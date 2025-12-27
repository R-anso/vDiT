# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

# [new]
from typing import List, Optional, Dict
from SAM.attn_utils.attn_tools import Attn_Info, Attn_Save_Cfg
from SAM.attn_utils.attn_tools import save_score
from SAM.attn_utils.recover import ClusterRecover

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
        cluster_recover: Optional[ClusterRecover] = None,
        layer_idx: Optional[int] = None,
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

        # [New] Blueprint Recovery Logic
        if cluster_recover is not None and layer_idx is not None:
            try:
                # 准备输出容器 [B, S, N, D]
                recovered_x = torch.zeros_like(q)
                
                # 遍历 Batch
                for i in range(b):
                    curr_len = int(seq_lens[i].item())
                    if curr_len == 0: continue
                    
                    # 获取当前样本的维度信息
                    curr_grid = grid_sizes[i].tolist() # [F, H, W]
                    l_f, l_h, l_w = curr_grid[0], curr_grid[1], curr_grid[2]
                    
                    # 切片获取当前样本的 Q, K, V (使用 RoPE 后的 Q, K)
                    q_sample = q_rot[i, :curr_len] # [L_i, N, D]
                    k_sample = k_rot[i, :curr_len]
                    v_sample = v[i, :curr_len]
                    
                    # 遍历 Heads
                    for h_idx in range(n):
                        # 1. 加载蓝图 (从内存缓存)
                        cluster_info = cluster_recover.load_blueprint(layer_idx, h_idx)
                        
                        # 2. 准备数据
                        q_h = q_sample[:, h_idx, :] # [L_i, D]
                        k_h = k_sample[:, h_idx, :]
                        v_h = v_sample[:, h_idx, :]
                        
                        # 3. 准备恢复所需中间变量
                        prepared_data = cluster_recover.prepare_data(
                            q_h, k_h, cluster_info, l_h, l_w, l_f
                        )
                        
                        # 4. 恢复并计算 Attention * V
                        out_h = cluster_recover.recover_matrix(
                            *prepared_data, v_h, cluster_info, l_h, l_w, l_f
                        ) # [L_i, D]
                        
                        # 填入结果
                        recovered_x[i, :curr_len, h_idx, :] = out_h

                # 如果成功完成所有 Batch 和 Head，则直接输出
                x = recovered_x.flatten(2)
                x = self.o(x)
                return x

            except Exception as e:
                # 如果恢复过程中出现任何错误（如缺少蓝图、维度不匹配等），回退到 Flash Attention
                print(f"[WanSelfAttention] Recovery failed for L{layer_idx}: {e}. Fallback to Flash Attention.")
                pass

        x = flash_attention(
            q=q_rot,
            k=k_rot,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        if attn_info is not None:
            self._compute_and_save_attn(q_rot, k_rot, seq_lens, attn_info)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    # [new]
    def _compute_and_save_attn(self, q, k, seq_lens, attn_info):
        block_size = attn_info.B_hw or 0
        if block_size <= 0 or not attn_info.head_idx:
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

            for head_idx in range(self.num_heads):
                if not attn_info.check_save_en(head_idx):
                    continue

                rope_order = attn_info.get_rope_order()
                use_components = bool(rope_order)
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
                    q_chunk = q_slice[row_start:row_end, head_idx]

                    col_start = 0
                    while col_start < target_len:
                        col_end = min(col_start + block_size, target_len)
                        k_chunk = k_slice[col_start:col_end, head_idx]

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

                fname_template = attn_info.get_fname(this_head=head_idx)
                if use_components:
                    tokens = [tok for tok in rope_order.split("-") if tok]
                    stacked = torch.stack([component_buffers[token] for token in tokens], dim=0)
                    save_score(
                        stacked,
                        out_dir=out_dir,
                        out_fmt=attn_info.out_fmt,
                        fname_template=fname_template,
                        rope_order=rope_order,
                    )
                else:
                    save_score(
                        attn_matrix,
                        out_dir=out_dir,
                        out_fmt=attn_info.out_fmt,
                        fname_template=fname_template,
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
        cluster_recover: Optional[ClusterRecover] = None,
        layer_idx: Optional[int] = None,
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
            seq_lens,
            grid_sizes,
            freqs,
            attn_info,
            cluster_recover=cluster_recover,
            layer_idx=layer_idx,
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
                 blueprint_dir: Optional[str] = None,
                 recover_start_iter: int = 10,
                 parallelism: int = 1024,
                 mag_en: bool = False):
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

        # [New] Initialize Recover Managers
        self.recover_start_iter = recover_start_iter
        self.recover_managers: Dict[str, ClusterRecover] = {}

        if blueprint_dir:
            import os
            for key in ['cond', 'uncond']:
                sub_dir = os.path.join(blueprint_dir, key)
                if os.path.exists(sub_dir):
                    print(f"[WanModel] Initializing ClusterRecover for '{key}' from {sub_dir}")
                    # preload_all=True to load all blueprints into CPU RAM
                    self.recover_managers[key] = ClusterRecover(sub_dir, recover_enabled=True, preload_all=True, parallelism=parallelism, mag_en=mag_en)
                else:
                    print(f"[WanModel] Warning: Blueprint subdir {sub_dir} not found.")

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
            
            # [New] Select Recover Manager
            cluster_recover = None
            if (self.recover_managers 
                and iter_idx is not None 
                and iter_idx >= self.recover_start_iter
                and key in self.recover_managers):
                cluster_recover = self.recover_managers[key]

            x = block(
                x,
                attn_info=attn_info,
                **kwargs,
                cluster_recover=cluster_recover,
                layer_idx=layer_idx,
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
            or not attn_save_cfg.enable
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
        )
