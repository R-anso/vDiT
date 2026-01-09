# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

# [new]
from typing import List, Optional, Dict
from SAM.attn_utils.attn_tools import Attn_Info, Attn_Save_Cfg, save_tensor
from SAM.attn_utils.attn_QK_cluster import Attn_Cluster_Manager

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        half_dim = self.head_dim // 2
        base_complex = half_dim // 3
        f_complex = half_dim - 2 * base_complex
        self._rope_component_dims = (
            2 * f_complex,
            2 * base_complex,
            2 * base_complex,
        )

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        attn_info: Optional[Attn_Info] = None,
        layer_idx: Optional[int] = None,
        iter_idx: Optional[int] = None,
        attn_cluster_manager: Optional[Attn_Cluster_Manager] = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            attn_info(Attn_Info | None): 当前层需要保存的注意力信息
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # [new]
        q_rot = rope_apply(q, grid_sizes, freqs)
        k_rot = rope_apply(k, grid_sizes, freqs)

        # [new] Attention Clustering (Step 4)
        if attn_cluster_manager is not None and attn_cluster_manager.is_active_iter(iter_idx):
            # 只有在活跃迭代步才执行聚类逻辑
            sizes = {'f': grid_sizes[0, 0].item(), 'h': grid_sizes[0, 1].item(), 'w': grid_sizes[0, 2].item()}
            cfg = attn_cluster_manager.cfg
            
            q_list, k_list = [], []
            q_res_list, k_res_list = [], []
            for h_idx in range(n):
                qh = q_rot[0, :, h_idx, :]
                kh = k_rot[0, :, h_idx, :]
                
                if attn_cluster_manager.should_update(iter_idx):
                    attn_cluster_manager.update_mask(layer_idx, h_idx, qh, kh, sizes)
                
                masks = attn_cluster_manager.get_masks(layer_idx, h_idx)
                if masks is not None:
                    q_mask, k_mask = masks['Q'], masks['K']
                    
                    q_out = attn_cluster_manager.apply_clustering(qh, sizes, q_mask, cfg.q_cf, cfg.q_ch, cfg.q_cw)
                    k_out = attn_cluster_manager.apply_clustering(kh, sizes, k_mask, cfg.k_cf, cfg.k_ch, cfg.k_cw)

                    if cfg.res_compensate:
                        qh_clustered, qh_res = q_out
                        kh_clustered, kh_res = k_out
                        q_res_list.append(qh_res)
                        k_res_list.append(kh_res)
                    else:
                        qh_clustered = q_out
                        kh_clustered = k_out
                    
                    q_ratio = q_mask.float().mean().item()
                    k_ratio = k_mask.float().mean().item()
                    attn_cluster_manager.record_metrics(layer_idx, h_idx, iter_idx, q_ratio, k_ratio)
                    
                    q_list.append(qh_clustered)
                    k_list.append(kh_clustered)
                else:
                    q_list.append(qh)
                    k_list.append(kh)
                    if cfg.res_compensate:
                        q_res_list.append(torch.zeros_like(qh))
                        k_res_list.append(torch.zeros_like(kh))
            
            q_rot = torch.stack(q_list, dim=1).unsqueeze(0)
            k_rot = torch.stack(k_list, dim=1).unsqueeze(0)
            if cfg.res_compensate:
                q_res = torch.stack(q_res_list, dim=1).unsqueeze(0)
                k_res = torch.stack(k_res_list, dim=1).unsqueeze(0)

        # Attention calculation
        if attn_cluster_manager is not None and attn_cluster_manager.is_active_iter(iter_idx) and cfg.res_compensate:
            # Step 5: Res Value compensate
            # 原理优化：将 (Q_orig @ K_base^T + Q_base @ K_res^T) 
            # 转化为单个拼接矩阵的注意力计算，从而使用 Flash Attention 节省显存
            
            # 1. 在 head_dim 维度(dim=-1)拼接
            # q_orig, q_base, k_base, k_res 形状均为 [B, L, N, D]
            q_combined = torch.cat([q_rot + q_res, q_rot], dim=-1) # [B, L, N, 2D]
            k_combined = torch.cat([k_rot, k_res], dim=-1)         # [B, L, N, 2D]
            
            # 2. 转换为 Flash Attention 要求的格式 [B, N, L, D_new]
            q_combined = q_combined.transpose(1, 2)
            k_combined = k_combined.transpose(1, 2)
            v_t = v.transpose(1, 2) # [B, N, L, D]
            
            # 3. 使用 PyTorch 内置的闪电注意力（自动选择 FlashAttention 或 MemoryEfficientAttention）
            # 注意：scale 必须设为原来的 1/sqrt(d)，而不是 1/sqrt(2d)
            scale = 1.0 / math.sqrt(self.head_dim)
            x = torch.nn.functional.scaled_dot_product_attention(
                q_combined, 
                k_combined, 
                v_t, 
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=False, 
                scale=scale
            )
            
            # 4. 还原形状 [B, N, L, D] -> [B, L, N, D]
            x = x.transpose(1, 2)
        else:
            # Flash Attention 路径（默认或 Step 4）
            x = flash_attention(
                q=q_rot,
                k=k_rot,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
            )

        if attn_info is not None:
            self._compute_and_save_attn(q_rot, k_rot, seq_lens, attn_info)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    # [new]
    def _compute_and_save_attn(self, q, k, seq_lens, attn_info):
        # 1. 检查保存开关
        save_q = attn_info.should_save("q")
        save_k = attn_info.should_save("k")
        save_score = attn_info.should_save("score")

        if not (save_q or save_k or save_score):
            return
        
        if not attn_info.head_idx:
            return

        block_size = attn_info.B_hw or 0
        # 如果只存 Q/K，block_size 可以忽略；如果存 Score，block_size 必须 > 0
        if save_score and block_size <= 0:
            # 无法计算 Score，降级为只存 Q/K (如果启用)
            save_score = False
            if not (save_q or save_k):
                return

        scale = 1.0 / math.sqrt(self.head_dim)
        component_dims = self._rope_component_dims
        canonical_tokens = ("f", "h", "w")
        out_dir = attn_info.get_real_out_dir()

        for batch_idx in range(q.size(0)):
            target_len = int(seq_lens[batch_idx].item())
            if target_len <= 0:
                continue

            q_slice = q[batch_idx, :target_len]
            k_slice = k[batch_idx, :target_len]

            for head_idx in attn_info.head_idx:
                if not attn_info.check_save_en(head_idx):
                    continue

                rope_order = attn_info.get_rope_order()
                use_components = bool(rope_order)
                tokens = [tok for tok in rope_order.split("-") if tok] if use_components else []

                q_head = q_slice[:, head_idx]
                k_head = k_slice[:, head_idx]

                # --- 保存 Q ---
                if save_q:
                    fname_q = attn_info.get_fname(this_head=head_idx, name="Q")
                    if use_components:
                        # 传入原始 q_head 和 component_dims，让 save_tensor 内部处理分割
                        save_tensor(
                            q_head,
                            out_dir=out_dir,
                            out_fmt=attn_info.out_fmt,
                            fname_template=fname_q,
                            rope_order=rope_order,
                            component_dims=component_dims,
                        )
                    else:
                        save_tensor(
                            q_head,
                            out_dir=out_dir,
                            out_fmt=attn_info.out_fmt,
                            fname_template=fname_q,
                        )

                # --- 保存 K ---
                if save_k:
                    fname_k = attn_info.get_fname(this_head=head_idx, name="K")
                    if use_components:
                        # 传入原始 k_head 和 component_dims
                        save_tensor(
                            k_head,
                            out_dir=out_dir,
                            out_fmt=attn_info.out_fmt,
                            fname_template=fname_k,
                            rope_order=rope_order,
                            component_dims=component_dims,
                        )
                    else:
                        save_tensor(
                            k_head,
                            out_dir=out_dir,
                            out_fmt=attn_info.out_fmt,
                            fname_template=fname_k,
                        )

                # --- 保存 Score ---
                if save_score:
                    if use_components:
                        component_buffers = {
                            key: torch.empty((target_len, target_len), dtype=q.dtype, device="cpu")
                            for key in canonical_tokens
                        }
                    else:
                        attn_matrix = torch.empty(
                            (target_len, target_len), dtype=q.dtype, device="cpu"
                        )

                    row_start = 0
                    while row_start < target_len:
                        row_end = min(row_start + block_size, target_len)
                        q_chunk = q_head[row_start:row_end]

                        col_start = 0
                        while col_start < target_len:
                            col_end = min(col_start + block_size, target_len)
                            k_chunk = k_head[col_start:col_end]

                            if use_components:
                                q_f, q_h, q_w = torch.split(q_chunk, component_dims, dim=-1)
                                k_f, k_h, k_w = torch.split(k_chunk, component_dims, dim=-1)

                                blocks = {
                                    "f": torch.matmul(q_f, k_f.transpose(0, 1)) * scale,
                                    "h": torch.matmul(q_h, k_h.transpose(0, 1)) * scale,
                                    "w": torch.matmul(q_w, k_w.transpose(0, 1)) * scale,
                                }
                                for key, block in blocks.items():
                                    component_buffers[key][
                                        row_start:row_end, col_start:col_end
                                    ] = block.cpu()
                            else:
                                block_partial = torch.matmul(
                                    q_chunk, k_chunk.transpose(0, 1)
                                ) * scale
                                attn_matrix[row_start:row_end, col_start:col_end] = block_partial.cpu()

                            col_start = col_end
                        row_start = row_end

                    fname_score = attn_info.get_fname(this_head=head_idx, name="score")
                    if use_components:
                        # Score 矩阵维度一致，可以 stack，不需要 component_dims
                        stacked = torch.stack([component_buffers[token] for token in tokens], dim=0)
                        save_tensor(
                            stacked,
                            out_dir=out_dir,
                            out_fmt=attn_info.out_fmt,
                            fname_template=fname_score,
                            rope_order=rope_order,
                        )
                    else:
                        save_tensor(
                            attn_matrix,
                            out_dir=out_dir,
                            out_fmt=attn_info.out_fmt,
                            fname_template=fname_score,
                        )


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        attn_info: Optional[Attn_Info] = None,
        layer_idx: Optional[int] = None,
        iter_idx: Optional[int] = None,
        attn_cluster_manager: Optional[Attn_Cluster_Manager] = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            attn_info(Attn_Info | None): 当前层需要保存的注意力信息
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # [new] self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            attn_info=attn_info,
            layer_idx=layer_idx,
            iter_idx=iter_idx,
            attn_cluster_manager=attn_cluster_manager,
        )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        iter_idx: Optional[int] = None,
        key: Optional[str] = None,
        attn_save_cfg: Optional[Attn_Save_Cfg] = None,
        attn_cluster_manager: Optional[Attn_Cluster_Manager] = None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            iter_idx (`int`, *optional*):
                当前扩散迭代步，用于判定是否需要保存注意力
            key (`str`, *optional*):
                “cond/uncond”等标签，用于区分CFG分支
            attn_save_cfg (Attn_Save_Cfg, *optional*):
                注意力保存配置
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        # [new]
        for layer_idx, block in enumerate(self.blocks):
            attn_info = self._build_attn_info(
                attn_save_cfg=attn_save_cfg,
                key=key,
                iter_idx=iter_idx,
                layer_idx=layer_idx,
            )

            x = block(
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                attn_info=attn_info,
                layer_idx=layer_idx,
                iter_idx=iter_idx,
                attn_cluster_manager=attn_cluster_manager,
            )

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    # [new]
    def _build_attn_info(
        self,
        attn_save_cfg: Optional[Attn_Save_Cfg],
        key: Optional[str],
        iter_idx: Optional[int],
        layer_idx: int,
    ) -> Optional[Attn_Info]:
        """若配置允许当前层保存注意力，则构造 Attn_Info"""
        if (
            attn_save_cfg is None
            or not attn_save_cfg.any_enabled()
            or key is None
            or iter_idx is None
            or not attn_save_cfg.check_save_en(key, iter_idx, layer_idx)
        ):
            return None

        head_list = list(attn_save_cfg.get_head_list() or [])
        if not head_list:
            return None

        return Attn_Info(
            key=key,
            iter_idx=iter_idx,
            layer_idx=layer_idx,
            head_idx=head_list,
            B_hw=attn_save_cfg.B_hw,
            out_dir=attn_save_cfg.out_dir,
            out_fmt=attn_save_cfg.out_fmt,
            rope_order=attn_save_cfg.get_rope_order(),
            enable_map=attn_save_cfg.get_enable_map(),
        )
