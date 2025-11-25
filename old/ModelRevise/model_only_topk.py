# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

import os
from .attention_utils import compare_and_save_scores
import torch.nn.functional as F
from SAM import SA_config
from SAM.mask_gen_ndiff import generate_diff_sparse_mask
from SAM.mask_eva import mask_eva

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

        # counter
        self.register_buffer('counter', torch.zeros(1, dtype=torch.uint8))

    def forward(self, x, seq_lens, grid_sizes, freqs, layer_idx=None, attn_save=None, sparse_mask_info=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q_rope = rope_apply(q, grid_sizes, freqs)
        k_rope = rope_apply(k, grid_sizes, freqs)

        scores = None
        q_perm = k_perm = None

        def ensure_permuted():
            nonlocal q_perm, k_perm
            if q_perm is None:
                q_perm = q_rope.permute(0, 2, 1, 3).contiguous()
                k_perm = k_rope.permute(0, 2, 1, 3).contiguous()
            return q_perm, k_perm

        if attn_save is not None and attn_save.get('enable', False):
            try:
                q_perm, k_perm = ensure_permuted()
                head_idx = int(attn_save.get('head', 0))
                batch_idx = int(attn_save.get('batch', 0)) if 'batch' in attn_save else 0
                if head_idx < 0 or head_idx >= q_perm.size(1):
                    head_idx = 0
                q_h = q_perm[:, head_idx:head_idx + 1]
                k_h = k_perm[:, head_idx:head_idx + 1]
                scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(d)
                scores = F.softmax(scores, dim=-1)

                out_dir = attn_save.get('out_dir', './attn_analysis/attn_dump')
                param = float(attn_save.get('param', 0.01))
                step = int(attn_save.get('step', 0))
                mode = attn_save.get('mode', 'topk')
                block_parts = attn_save.get('block_parts', None)
                attn_mask_format = attn_save.get('out_format', 'npy')
                compare_and_save_scores(
                    scores,
                    param,
                    out_dir,
                    step,
                    layer_idx or 0,
                    head_idx,
                    batch_idx,
                    mode,
                    block_parts=block_parts,
                    out_format=attn_mask_format,
                )
            except Exception:
                pass

        if not SA_config.config.get('enabled', False):
            x = flash_attention(
                q=q_rope,
                k=k_rope,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)
        else:
            cfg = SA_config.config
            q_perm, k_perm = ensure_permuted()
            scale = 1.0 / math.sqrt(d)
            A = torch.matmul(q_perm, k_perm.transpose(-2, -1)) * scale

            self.counter += 1
            current_layer_idx = layer_idx or 0
            B_hw = cfg['B_hw']
            num_block_sqrt = (s + B_hw - 1) // B_hw
            device = A.device

            final_mask = torch.empty((b, n, s, s), dtype=torch.bool, device=device)
            evaluate_masks = cfg.get('evaluate_mask', False)
            all_head_A = [] if evaluate_masks else None

            total_blocks = b * num_block_sqrt * num_block_sqrt
            for head_idx in range(n):
                A_head = A[:, head_idx, :, :]
                final_mask_head = final_mask[:, head_idx]
                final_mask_head.zero_()

                if evaluate_masks:
                    all_head_A.append(A_head.detach())

                blocks_buffer = torch.empty((total_blocks, B_hw, B_hw), dtype=A.dtype, device=device)
                block_ptr = 0
                for batch_idx in range(b):
                    for row in range(num_block_sqrt):
                        start_row = row * B_hw
                        end_row = min((row + 1) * B_hw, s)
                        row_extent = end_row - start_row
                        for col in range(num_block_sqrt):
                            start_col = col * B_hw
                            end_col = min((col + 1) * B_hw, s)
                            col_extent = end_col - start_col
                            block_view = blocks_buffer[block_ptr]
                            block_view.zero_()
                            block_view[:row_extent, :col_extent].copy_(
                                A_head[batch_idx, start_row:end_row, start_col:end_col])
                            block_ptr += 1

                generated_masks = generate_diff_sparse_mask(
                    blocks_buffer,
                    B_h=cfg['B_h'],
                    B_w=cfg['B_w'],
                    alpha=cfg['alpha'],
                    beta=cfg['beta'],
                    log_enabled=cfg['log_enabled'],
                    log_dir=cfg['log_dir'],
                    mode=cfg['ndiff_mode'],
                ).view(b, num_block_sqrt, num_block_sqrt, B_hw, B_hw)

                for batch_idx in range(b):
                    full_mask_view = final_mask_head[batch_idx]
                    for row in range(num_block_sqrt):
                        start_row = row * B_hw
                        end_row = min((row + 1) * B_hw, s)
                        row_extent = end_row - start_row
                        for col in range(num_block_sqrt):
                            start_col = col * B_hw
                            end_col = min((col + 1) * B_hw, s)
                            col_extent = end_col - start_col
                            mask_block = generated_masks[batch_idx, row, col]
                            full_mask_view[start_row:end_row, start_col:end_col].copy_(
                                mask_block[:row_extent, :col_extent])

            if evaluate_masks:
                for head_idx in range(n):
                    mask_eva(
                        all_head_A[head_idx],
                        final_mask[:, head_idx, :, :],
                        log_enabled=True,
                        log_dir=os.path.join(cfg['log_dir'], f"layer_{current_layer_idx}", f"head_{head_idx}"),
                        iter_step=(self.counter / 2 + 1),
                    )

            masked_scores = A.masked_fill(~final_mask, -float('inf'))
            attn_probs = F.softmax(masked_scores, dim=-1)

            v_perm = v.permute(0, 2, 1, 3)
            x = torch.matmul(attn_probs, v_perm)
            x = x.permute(0, 2, 1, 3)

        x = x.flatten(2)
        x = self.o(x)
        return x


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
        layer_idx=None,
        attn_save=None,
        sparse_mask_info=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        # y = self.self_attn(
        #     self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
        #     seq_lens, grid_sizes, freqs)
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, layer_idx, attn_save, sparse_mask_info)
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
                 eps=1e-6):
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

        # 为稀疏注意力初始化状态
        if SA_config.config.get('enabled', False):
            # 使用 register_buffer 将其注册为模型状态，但 non-persistent
            self.register_buffer('sparse_mask_past', None, persistent=False)
            self.register_buffer('sparse_timesteps_past', None, persistent=False)
        self._sparse_histories = {}

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        attn_save=None,
        sparse_history_key: str = "default",
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
        
        batch_size = len(x)

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

        # 准备稀疏掩码所需的信息
        sparse_mask_info = None
        history_state = None
        sparse_mask_info = None
        if SA_config.config.get('enabled', False):
            sparse_mask_info = {}

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

        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                layer_idx=layer_idx,
                attn_save=attn_save,
                sparse_mask_info=sparse_mask_info,
                **kwargs,
            )

        # head
        x = self.head(x, e)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def reset_sparse_history(self, keys=None):
        """
        清空稀疏注意力历史缓存。

        Args:
            keys (Iterable[str] or str, optional): 指定需要清理的分支；默认清除全部。
        """
        if not hasattr(self, "_sparse_histories"):
            self._sparse_histories = {}
            return
        if keys is None:
            self._sparse_histories.clear()
            return
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            self._sparse_histories.pop(key, None)

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
