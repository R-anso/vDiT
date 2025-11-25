# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

import os
from attn_utils.save_mask import save_mask
from attn_utils.save_score import save_score
import torch.nn.functional as F
from SAM import SA_config
from SAM.mask_gen_ndiff import generate_diff_sparse_mask
from SAM.mask_gen_ndiff_lm import generate_diff_sparse_mask_lm
from SAM.mask_iter import predict_sparse_attn_mask
from SAM.mask_iter_lm import predict_sparse_attn_mask_lm
from SAM.mask_eva import mask_eva
from SAM.mask_eva_lm import mask_eva_lm

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

    # def forward(self, x, seq_lens, grid_sizes, freqs):
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
            q_perm, k_perm = ensure_permuted()
            head_idx = int(attn_save.get('head', 0))
            batch_idx = int(attn_save.get('batch', 0)) if 'batch' in attn_save else 0
            if head_idx < 0 or head_idx >= q_perm.size(1):
                head_idx = 0
            q_h = q_perm[:, head_idx:head_idx + 1]
            k_h = k_perm[:, head_idx:head_idx + 1]
            scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(d)
            scores = F.softmax(scores, dim=-1)

            out_dir = attn_save.get('mask_out_dir', './attn_analysis/attn_mask')
            param = float(attn_save.get('param', 0.01))
            step = int(attn_save.get('step', 0))
            mode = attn_save.get('mode', 'topk')
            block_parts = attn_save.get('block_parts', None)
            attn_mask_format = attn_save.get('out_format', 'npy')
            save_mask(
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
            if attn_save.get('score_save_enable', False):
                save_score(
                    scores,
                    out_dir=attn_save.get('score_out_dir', './attn_analysis/attn_score'),
                    step=int(attn_save.get('step', 0)),
                    layer_idx=layer_idx or 0,
                    head_idx=int(attn_save.get('head', 0)),
                    out_format=attn_mask_format,
                )

        if not SA_config.config.get('enabled', False) or sparse_mask_info is None:
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
            # 优化：不再预先计算巨大的完整矩阵 A
            # A = torch.matmul(q_perm, k_perm.transpose(-2, -1)) * scale

            self.counter += 1
            current_layer_idx = layer_idx or 0
            timesteps_past_layer = sparse_mask_info['timesteps_past'][current_layer_idx]
            mask_past_layer = sparse_mask_info['mask_past'][current_layer_idx]
            updated_storage = sparse_mask_info.get('updated_mask_past') if sparse_mask_info else None

            if isinstance(timesteps_past_layer, (list, tuple)):
                valid_history_count = sum(1 for ts in timesteps_past_layer if ts != 0)
            else:
                valid_history_count = int(torch.count_nonzero(timesteps_past_layer).item())
            bootstrap_mode = valid_history_count < cfg['num_past_step']

            B_hw = cfg['B_hw']
            num_block_sqrt = (s + B_hw - 1) // B_hw
            device = q_perm.device
            final_mask = torch.empty((b, n, s, s), dtype=torch.uint8, device=device)
            evaluate_masks = cfg.get('evaluate_mask', False)
            # 优化：不再存储所有头的 A 用于评估，改为即时评估
            # all_head_A = [] if evaluate_masks else None

            # 预先准备 V 的排列，以便在循环中使用
            v_perm = v.permute(0, 2, 1, 3)
            x_heads = []

            for head_idx in range(n):
                # 优化：逐头计算 Attention Score，大幅降低显存峰值
                q_head = q_perm[:, head_idx] # [B, S, D]
                k_head = k_perm[:, head_idx] # [B, S, D]
                A_head = torch.matmul(q_head, k_head.transpose(-2, -1)) * scale # [B, S, S]

                final_mask_head = final_mask[:, head_idx]
                final_mask_head.zero_()
                # if evaluate_masks:
                #     all_head_A.append(A_head.detach())

                if bootstrap_mode:
                    total_blocks = b * (num_block_sqrt ** 2)
                    blocks_buffer = torch.empty((total_blocks, B_hw, B_hw), device=device, dtype=A_head.dtype)
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

                    if cfg['low_memory_mode']:
                        generated_masks_batched = generate_diff_sparse_mask_lm(
                            blocks_buffer,
                            B_h=cfg['B_h'],
                            B_w=cfg['B_w'],
                            alpha=cfg['alpha'],
                            beta=cfg['beta'],
                            mode=cfg['ndiff_mode'],
                        ).to(device=device, dtype=torch.uint8).view(
                            b, num_block_sqrt * num_block_sqrt, B_hw, B_hw)
                    else:
                        generated_masks_batched = generate_diff_sparse_mask(
                            blocks_buffer,
                            B_h=cfg['B_h'],
                            B_w=cfg['B_w'],
                            alpha=cfg['alpha'],
                            beta=cfg['beta'],
                            log_enabled=cfg['log_enabled'],
                            log_dir=cfg['log_dir'],
                            mode=cfg['ndiff_mode'],
                        ).to(device=device, dtype=torch.uint8).view(
                            b, num_block_sqrt * num_block_sqrt, B_hw, B_hw)

                    for batch_idx in range(b):
                        full_mask_view = final_mask_head[batch_idx]
                        full_mask_view.zero_()
                        dest_last = None
                        if updated_storage is not None:
                            dest = updated_storage[current_layer_idx, batch_idx, head_idx]
                            if dest.shape[0] > 1:
                                dest[:-1].copy_(mask_past_layer[batch_idx, head_idx][1:])
                            dest_last = dest[-1]
                            dest_last.zero_()
                        for row in range(num_block_sqrt):
                            start_row = row * B_hw
                            end_row = min((row + 1) * B_hw, s)
                            row_extent = end_row - start_row
                            for col in range(num_block_sqrt):
                                start_col = col * B_hw
                                end_col = min((col + 1) * B_hw, s)
                                col_extent = end_col - start_col
                                mask_block = generated_masks_batched[batch_idx, row * num_block_sqrt + col]
                                full_mask_view[start_row:end_row, start_col:end_col].copy_(
                                    mask_block[:row_extent, :col_extent])
                                if dest_last is not None and row == col:
                                    dest_block = dest_last[row]
                                    dest_block[:row_extent, :col_extent].copy_(
                                        mask_block[:row_extent, :col_extent].to(dtype=dest_block.dtype))
                else:
                    total_diag_blocks = b * num_block_sqrt
                    diag_blocks_buffer = torch.empty((total_diag_blocks, B_hw, B_hw), device=device, dtype=A_head.dtype)
                    block_ptr = 0
                    for batch_idx in range(b):
                        for row in range(num_block_sqrt):
                            start = row * B_hw
                            end = min((row + 1) * B_hw, s)
                            extent = end - start
                            block_view = diag_blocks_buffer[block_ptr]
                            block_view.zero_()
                            block_view[:extent, :extent].copy_(A_head[batch_idx, start:end, start:end])
                            block_ptr += 1

                    if cfg['low_memory_mode']:
                        mask_diag_now_batched = generate_diff_sparse_mask_lm(
                            diag_blocks_buffer,
                            B_h=cfg['B_h'],
                            B_w=cfg['B_w'],
                            alpha=cfg['alpha'],
                            beta=cfg['beta'],
                            mode=cfg['ndiff_mode'],
                        ).to(device=device, dtype=torch.uint8).view(b, num_block_sqrt, B_hw, B_hw)
                    else:
                        mask_diag_now_batched = generate_diff_sparse_mask(
                            diag_blocks_buffer,
                            B_h=cfg['B_h'],
                            B_w=cfg['B_w'],
                            alpha=cfg['alpha'],
                            beta=cfg['beta'],
                            log_enabled=cfg['log_enabled'],
                            log_dir=cfg['log_dir'],
                            mode=cfg['ndiff_mode'],
                        ).to(device=device, dtype=torch.uint8).view(b, num_block_sqrt, B_hw, B_hw)

                    for batch_idx in range(b):
                        mask_diag_now = mask_diag_now_batched[batch_idx]
                        mask_past_head = mask_past_layer[batch_idx, head_idx]
                        if cfg['low_memory_mode']:
                            full_mask_pred, updated_mask_past = predict_sparse_attn_mask_lm(
                                B_hw=B_hw,
                                num_block_sqrt=num_block_sqrt,
                                mask_past=mask_past_head,
                                timesteps_past=timesteps_past_layer,
                                timestep_now=sparse_mask_info['timestep_now'],
                                alpha=cfg['alpha_iter'],
                                threshold_inter=cfg['threshold_inter'],
                                mask_diag_now=mask_diag_now,
                                d_disappear=cfg['d_disappear'],
                                num_past_step=cfg['num_past_step'],
                                dist_func=cfg['dist_func'],
                            )
                        else:
                            full_mask_pred, updated_mask_past = predict_sparse_attn_mask(
                                B_hw=B_hw,
                                num_block_sqrt=num_block_sqrt,
                                mask_past=mask_past_head,
                                timesteps_past=timesteps_past_layer,
                                timestep_now=sparse_mask_info['timestep_now'],
                                alpha=cfg['alpha_iter'],
                                threshold_inter=cfg['threshold_inter'],
                                mask_diag_now=mask_diag_now,
                                d_disappear=cfg['d_disappear'],
                                num_past_step=cfg['num_past_step'],
                                dist_func=cfg['dist_func'],
                                log_enabled=cfg['log_enabled'],
                                log_dir=cfg['log_dir'],
                            )
                        final_mask_head[batch_idx].copy_(full_mask_pred)
                        if updated_storage is not None:
                            updated_storage[current_layer_idx, batch_idx, head_idx].copy_(updated_mask_past)

                # 优化：即时评估，避免存储 all_head_A
                if evaluate_masks and head_idx == 0:
                    # for head_idx in range(n):
                    # head_idx = 0
                    if cfg['ndiff_mode']:
                        mask_eva_lm(
                            A_head.detach(),
                            final_mask_head,
                            log_enabled=True,
                            log_dir=os.path.join(cfg['log_dir'], f"layer_{current_layer_idx}", f"head_{head_idx}"),
                            iter_step=(self.counter / 2 + 1),
                        )
                    else:
                        mask_eva(
                            A_head.detach(),
                            final_mask_head,
                            log_enabled=True,
                            log_dir=os.path.join(cfg['log_dir'], f"layer_{current_layer_idx}", f"head_{head_idx}"),
                            iter_step=(self.counter / 2 + 1),
                        )

                # 优化：即时应用 Mask 并计算输出，随后释放 A_head
                masked_scores_head = A_head.masked_fill(final_mask_head == 0, -float('inf'))
                attn_probs_head = F.softmax(masked_scores_head, dim=-1)
                
                # v_perm: [B, N, S, D] -> v_head: [B, S, D]
                v_head = v_perm[:, head_idx]
                
                # x_head: [B, S, D]
                x_head = torch.matmul(attn_probs_head, v_head)
                x_heads.append(x_head)
                
                # 显式删除大张量
                del A_head, masked_scores_head, attn_probs_head

            # 重组输出 x
            # x_heads 是 [B, S, D] 的列表 -> stack 为 [B, N, S, D]
            x = torch.stack(x_heads, dim=1)
            # 恢复原始流程的维度 [B, S, N, D]
            x = x.permute(0, 2, 1, 3)

            if cfg.get('attn_save_enabled', False):
                # for head_idx in range(n):
                save_mask(
                    scores=None,
                    param=None,
                    out_dir=cfg['attn_save_dir'],
                    step=self.counter.item(),
                    layer_idx=current_layer_idx,
                    head_idx=0,
                    batch_idx=batch_idx,
                    mode="topk",
                    block_parts=None,
                    out_format=cfg['attn_save_format'],
                    mask_gen=False,
                    mask=final_mask,
                )
            # attn_probs = F.softmax(masked_scores, dim=-1)
            # attn_probs = F.softmax(masked_scores, dim=-1)

            # v_perm = v.permute(0, 2, 1, 3)
            # x = torch.matmul(attn_probs, v_perm)
            # x = x.permute(0, 2, 1, 3)

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
        if SA_config.config.get('enabled', False):
            cfg = SA_config.config
            num_past_step = cfg['num_past_step']
            B_hw = cfg['B_hw']
            num_block_sqrt = (seq_len + B_hw - 1) // B_hw
            timestep_now = int(t.reshape(-1)[0].item())
            expected_mask_shape = (
                self.num_layers,
                batch_size,
                self.num_heads,
                num_past_step,
                num_block_sqrt,
                B_hw,
                B_hw,
            )
            expected_time_shape = (self.num_layers, num_past_step)

            state = self._sparse_histories.get(sparse_history_key)
            if (
                state is None
                or state['mask'].shape != expected_mask_shape
                or state['timesteps'].shape != expected_time_shape
            ):
                state = {
                    'mask': torch.zeros(
                        expected_mask_shape, device='cpu', dtype=torch.uint8
                    ),
                    'timesteps': torch.zeros(
                        expected_time_shape, device='cpu', dtype=torch.long
                    ),
                }
                self._sparse_histories[sparse_history_key] = state

            # 仅在当前前向中把历史搬到 GPU
            mask_history_gpu = state['mask'].to(device=device, non_blocking=True)
            timesteps_history_gpu = state['timesteps'].to(device=device, non_blocking=True)

            history_state = state
            updated_buffer = torch.empty_like(mask_history_gpu)
            sparse_mask_info = {
                'mask_past': mask_history_gpu,
                'timesteps_past': timesteps_history_gpu.detach(),
                'timestep_now': timestep_now,
                'updated_mask_past': updated_buffer,
            }

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

        # for block in self.blocks:
        #     x = block(x, **kwargs)
        # 逐层传入 layer_idx 与 attn_save（如果上层提供）
        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                layer_idx=layer_idx,
                attn_save=attn_save,
                sparse_mask_info=sparse_mask_info,
                **kwargs,
            )

        # 更新稀疏掩码状态
        if sparse_mask_info is not None:
            history_state = self._sparse_histories[sparse_history_key]
            history_state['mask'].copy_(sparse_mask_info['updated_mask_past'].to('cpu'))

            timestep_now_tensor = torch.full(
                (self.num_layers, 1),
                sparse_mask_info['timestep_now'],
                device=device,
                dtype=history_state['timesteps'].dtype,
            )
            if history_state['timesteps'].size(1) > 1:
                shifted = torch.cat(
                    [history_state['timesteps'].to(device)[:, 1:], timestep_now_tensor], dim=1
                )
            else:
                shifted = timestep_now_tensor
            history_state['timesteps'].copy_(shifted.to('cpu'))

            # 及时释放 GPU 上的中间缓存
            del sparse_mask_info

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
