# SFCDC version: 5.1 (Performance-optimized)
# Key optimizations over v5.0:
#   1. GPU-resident centroids (no CPU round-trip, no CUDA sync)
#   2. No chunk loop — all B*H heads processed in one batch
#   3. batch_kmeans_Euclid for both init & step (fused assign+update)
#   4. Triton permute/inverse-permute kernels (replaces 3x argsort + advanced indexing)
#   5. cluster_sizes reused from kmeans output (no redundant scatter_add)

import torch
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from svg.ops.sfcdc_triton import sfcdc_attention_v4_2
from svg.kmeans_utils import batch_kmeans_Euclid
from svg.kernels.triton.permute import permute_tensor_by_labels_triton, apply_inverse_permutation_triton


class SFCDC_Simulator:
    def __init__(self, enabled=False, clusbegin_iter=11, start_iter=12, end_iter=46,
                 centers_q=256, centers_k=256, group_size=4,
                 k0=0.25, k1=0.25, k2=0.50, l_f=16, l_h=16, l_w=16,
                 kmeans_iter_init=5, kmeans_iter_step=1,
                 **kwargs):
        self.enabled = enabled
        self.start_iter = start_iter
        self.clusbegin_iter = start_iter - 1 if start_iter > 0 else 0
        self.end_iter = end_iter
        self.centers_q = centers_q
        self.centers_k = centers_k
        self.group_size = group_size
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2

        self.l_f = l_f
        self.l_h = l_h
        self.l_w = l_w
        self.kmeans_iter_init = kmeans_iter_init
        self.kmeans_iter_step = kmeans_iter_step

        # GPU centroid cache: (key, layer_idx) -> {"q": [B_all, K, D], "k": [B_all, K, D]}
        self.gpu_centroids: Dict[Tuple, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _get_init_centroids(self, meta_key: Tuple, B_all: int):
        """Return cached GPU centroids if shape matches, else (None, None)."""
        prev = self.gpu_centroids.get(meta_key)
        if prev is not None and prev["q"].shape[0] == B_all:
            return prev["q"], prev["k"]
        return None, None

    # ------------------------------------------------------------------
    #  Main entry
    # ------------------------------------------------------------------
    def analyze(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                layer_idx: int, iter_idx: int, key: str) -> Optional[torch.Tensor]:
        """
        Main entry point for SFCDC Attention (v5.1).
        q, k, v: [B, L, N_Heads, D]
        """
        if not self.enabled or iter_idx < self.clusbegin_iter or iter_idx > self.end_iter:
            return None

        B, L, N, D = q.shape
        F_dim, H_dim, W_dim = self.l_f, self.l_h, self.l_w
        total_tokens = min(L, F_dim * H_dim * W_dim)
        device, dtype = q.device, q.dtype

        # Flatten Batch and Heads: [B*N, L, D]
        q_all = q.transpose(1, 2).reshape(-1, L, D).contiguous()
        k_all = k.transpose(1, 2).reshape(-1, L, D).contiguous()
        v_all = v.transpose(1, 2).reshape(-1, L, D).contiguous()

        if total_tokens < L:
            q_all = q_all[:, :total_tokens]
            k_all = k_all[:, :total_tokens]
            v_all = v_all[:, :total_tokens]
            curr_L = total_tokens
        else:
            curr_L = L

        B_all = q_all.size(0)
        meta_key = (key, layer_idx)

        # ---- Phase 1: Warmup — build centroids only ----
        if iter_idx < self.start_iter:
            init_cq, init_ck = self._get_init_centroids(meta_key, B_all)
            iters = self.kmeans_iter_step if init_cq is not None else self.kmeans_iter_init

            _, centroids_q, _, _ = batch_kmeans_Euclid(
                q_all, self.centers_q, max_iters=iters, init_centroids=init_cq)
            _, centroids_k, _, _ = batch_kmeans_Euclid(
                k_all, self.centers_k, max_iters=iters, init_centroids=init_ck)

            self.gpu_centroids[meta_key] = {
                "q": centroids_q.detach(),
                "k": centroids_k.detach(),
            }
            return None

        # ---- Phase 2: Inference with tiered sparse attention ----
        init_cq, init_ck = self._get_init_centroids(meta_key, B_all)

        # KMeans step (warm-start 1-iter: fused assign + centroid update)
        c_LQ, QC, qc_sizes, _ = batch_kmeans_Euclid(
            q_all, self.centers_q,
            max_iters=self.kmeans_iter_step, init_centroids=init_cq,
        )
        c_LK, KC, kc_sizes, _ = batch_kmeans_Euclid(
            k_all, self.centers_k,
            max_iters=self.kmeans_iter_step, init_centroids=init_ck,
        )

        self.gpu_centroids[meta_key] = {"q": QC.detach(), "k": KC.detach()}

        inv_sqrt_d = 1.0 / math.sqrt(D)

        # ---- Tier Table ----
        sc = torch.bmm(QC, KC.transpose(1, 2)) * inv_sqrt_d
        if torch.isnan(sc).any() or torch.isinf(sc).any():
            sc = torch.nan_to_num(sc, nan=0.0, posinf=1e4, neginf=-1e4)

        sc_idx = torch.sort(sc, dim=-1, descending=True)[1]
        c0 = int(self.k0 * self.centers_k)
        c1 = int(self.k1 * self.centers_k)
        c2 = int(self.k2 * self.centers_k)

        tier_table = torch.zeros_like(sc, dtype=torch.int32)
        tier_table.scatter_(2, sc_idx[:, :, :c0], 1)              # Tier 1: Exact
        tier_table.scatter_(2, sc_idx[:, :, c0:c0+c1], 2)         # Tier 2: Compressed
        tier_table.scatter_(2, sc_idx[:, :, c0+c1:c0+c1+c2], 3)   # Tier 3: Centroid

        # ---- Permute Q/K/V using Triton kernels  ----
        # permute_tensor_by_labels_triton expects [B, H, S, D] with dim=2
        q_4d = q_all.unsqueeze(1)   # [B_all, 1, L, D]
        k_4d = k_all.unsqueeze(1)
        v_4d = v_all.unsqueeze(1)

        q_perm, q_sorted_idx = permute_tensor_by_labels_triton(q_4d, c_LQ, dim=2)
        k_perm, k_sorted_idx = permute_tensor_by_labels_triton(k_4d, c_LK, dim=2)
        v_perm, _            = permute_tensor_by_labels_triton(v_4d, c_LK, dim=2,
                                                               sorted_indices=k_sorted_idx)

        # Sorted query labels (ascending after permutation)
        lq_sorted = torch.sort(c_LQ, dim=-1).values   # [B_all, L]

        # ---- V_sums per cluster ----
        v_flat  = v_all.flatten(0, 1)   # [B_all*L, D]
        lk_flat = (c_LK + (torch.arange(B_all, device=device) * self.centers_k).unsqueeze(1)).flatten()
        v_sums_flat = torch.zeros(B_all * self.centers_k, D, device=device, dtype=dtype)
        v_sums_flat.index_add_(0, lk_flat, v_flat.to(dtype))
        v_sums = v_sums_flat.view(B_all, self.centers_k, D)

        # ---- Cluster offsets (reuse kc_sizes from batch_kmeans_Euclid) ----
        k_counts_hist = kc_sizes.to(torch.int32)                     # [B_all, centers_k]
        k_offsets = torch.cumsum(k_counts_hist, dim=1)               # cumulative end indices
        k_starts = torch.cat([
            torch.zeros(B_all, 1, device=device, dtype=torch.int32),
            k_offsets[:, :-1],
        ], dim=1)
        k_ends = k_offsets

        # ---- Pad to BLOCK_M=64 alignment (Triton kernel has no boundary mask) ----
        BLOCK_M = 64
        L_actual = curr_L
        pad_len = (BLOCK_M - L_actual % BLOCK_M) % BLOCK_M
        if pad_len > 0:
            q_perm    = F.pad(q_perm,    (0, 0, 0, pad_len))
            k_perm    = F.pad(k_perm,    (0, 0, 0, pad_len))
            v_perm    = F.pad(v_perm,    (0, 0, 0, pad_len))
            lq_sorted = F.pad(lq_sorted, (0, pad_len), value=0)

        # ---- Prepare kernel inputs  [B_all, 1, *, *] ----
        qc_in        = QC.unsqueeze(1).contiguous()
        kc_in        = KC.unsqueeze(1).contiguous()
        v_sums_in    = v_sums.unsqueeze(1).contiguous()
        k_counts_in  = k_counts_hist.unsqueeze(1).contiguous()
        lq_in        = lq_sorted.unsqueeze(1).to(torch.int32).contiguous()
        k_starts_in  = k_starts.unsqueeze(1).contiguous()
        k_ends_in    = k_ends.unsqueeze(1).contiguous()
        tier_table_in = tier_table.unsqueeze(1).to(torch.int32).contiguous()

        # ---- SFCDC Triton Attention ----
        out_sorted = sfcdc_attention_v4_2(
            q_perm.contiguous(), k_perm.contiguous(), v_perm.contiguous(),
            qc_in, kc_in,
            v_sums_in, k_counts_in,
            lq_in, k_starts_in, k_ends_in,
            tier_table_in,
            sm_scale=inv_sqrt_d,
        )

        # ---- Remove padding & inverse permute ----
        out_sorted = out_sorted[:, :, :L_actual, :]             # [B_all, 1, L, D]
        out = apply_inverse_permutation_triton(out_sorted, q_sorted_idx, dim=2)

        # ---- Reshape back: [B_all, 1, L, D] -> [B, L, N, D] ----
        out = out.squeeze(1)                                     # [B_all, L, D]
        return out.reshape(B, N, curr_L, D).transpose(1, 2)
