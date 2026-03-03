# SFCDC version: 5.0 (CogVideoX1.5-style, with text token support and centroid scaling)

import torch
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from svg.ops.sfcdc_triton import sfcdc_attention_v4_2

def compiled_cdist(x: torch.Tensor, y: torch.Tensor, max_batch: int = 16) -> torch.Tensor:
    """
    Compute Euclidean distance matrix between x and y.
    x: [B, N, D]
    y: [B, M, D]
    Returns: [B, N, M]
    Sub-batches along B dimension to limit peak GPU memory.
    """
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"compiled_cdist expects rank-3 tensors, got x={x.shape}, y={y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"compiled_cdist shape mismatch: x={x.shape}, y={y.shape}")

    dtype_orig = x.dtype
    B = x.shape[0]
    max_batch = max(int(max_batch), 1)

    if B <= max_batch:
        xi = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        yi = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y
        return torch.cdist(xi, yi, p=2).to(dtype_orig)

    N, M = x.shape[1], y.shape[1]
    result = torch.empty(B, N, M, device=x.device, dtype=dtype_orig)
    for i in range(0, B, max_batch):
        j = min(i + max_batch, B)
        xi = x[i:j].float() if x.dtype in (torch.float16, torch.bfloat16) else x[i:j]
        yi = y[i:j].float() if y.dtype in (torch.float16, torch.bfloat16) else y[i:j]
        result[i:j] = torch.cdist(xi, yi, p=2).to(dtype_orig)
    return result

def batched_kmeans(x: torch.Tensor, num_clusters: int, num_iters: int = 5, init_centroids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched K-Means implementation optimized for GPU.
    """
    B, N, D = x.shape
    device = x.device
    
    # Handle edge case where sequence length is smaller than number of clusters
    if N < num_clusters:
        labels = torch.arange(N, device=device).unsqueeze(0).expand(B, -1) # [B, N]
        centroids = F.pad(x, (0, 0, 0, num_clusters - N), value=0)
        counts = torch.ones(B, num_clusters, device=device, dtype=x.dtype)
        counts[:, N:] = 0
        return labels, centroids, counts

    if init_centroids is None:
        # Optimized initialization: Randomly select points as centroids
        rand_cols = torch.randint(0, N, (B, num_clusters), device=device)
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * N
        flat_indices = (rand_cols + batch_offsets).flatten()
        centroids = x.flatten(0, 1)[flat_indices].view(B, num_clusters, D)
    else:
        centroids = init_centroids

    labels_offset = (torch.arange(B, device=device) * num_clusters).unsqueeze(1)
    
    for _ in range(num_iters):
        dists = compiled_cdist(x, centroids)
        labels = torch.argmin(dists, dim=-1)
        
        # Fast update using index_add_
        x_flat = x.flatten(0, 1)
        labels_flat = (labels + labels_offset).flatten()
        
        new_centroids_flat = torch.zeros(B * num_clusters, D, device=device, dtype=x.dtype)
        counts_flat = torch.zeros(B * num_clusters, device=device, dtype=x.dtype)
        
        new_centroids_flat.index_add_(0, labels_flat, x_flat)
        ones = torch.ones_like(labels_flat, dtype=x.dtype)
        counts_flat.index_add_(0, labels_flat, ones)
        
        new_centroids = new_centroids_flat.view(B, num_clusters, D)
        counts = counts_flat.view(B, num_clusters)
        
        # Avoid division by zero
        centroids = new_centroids / counts.clamp(min=1).unsqueeze(-1)
        
    return labels, centroids, counts
# -------------------------------------------

class SFCDC_Simulator:
    def __init__(self, enabled=False, clusbegin_iter=11, start_iter=12, end_iter=46, 
                 centers_q=256, centers_k=256, group_size=4, 
                 k0=0.25, k1=0.25, k2=0.50, l_f=16, l_h=16, l_w=16, 
                 q_quant_type='block', k_quant_type='token', dist_metric='L2', 
                 centroid_mode='mean', diff_mode='query',
                 text_token_length=0,
                 text_location='begin',
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
        
        # Fixed parameters for v4.2/4.3
        self.dist_metric = 'L2'           
        self.centroid_mode = 'mean'       
        
        self.l_f = l_f
        self.l_h = l_h
        self.l_w = l_w

        self.text_token_length = text_token_length
        assert text_location in ('begin', 'end'), f"text_location must be 'begin' or 'end', got '{text_location}'"
        self.text_location = text_location

        # State cache for centroids
        self.offline_meta: Dict[Tuple, Dict[str, torch.Tensor]] = {}

    def clear_cache(self):
        """Clear centroid cache and release GPU memory.
        Call this after denoising finishes (before VAE decoding) to free memory."""
        self.offline_meta.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def analyze(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                layer_idx: int, iter_idx: int, key: str) -> Optional[torch.Tensor]:
        """
        Main entry point for SFCDC Attention.
        q, k, v: [B, L, N_Heads, D]
        
        When text_token_length > 0:
          - text_location='begin' (default, CogVideoX): first T tokens = text
          - text_location='end' (HunyuanVideo): last T tokens = text
          - Text Q tokens form one always-tier-1 cluster
          - Text K tokens form one always-tier-1 cluster
          - Only video tokens are clustered by K-Means
          - The tier table is extended: video-video uses standard SFCDC tiering,
            text clusters are always tier 1 (exact computation)
        """
        if not self.enabled or iter_idx < self.clusbegin_iter or iter_idx > self.end_iter: 
            return None

        B, L, N, D = q.shape
        T = self.text_token_length
        F_dim, H_dim, W_dim = self.l_f, self.l_h, self.l_w
        device, dtype = q.device, q.dtype

        expected_len = F_dim * H_dim * W_dim + T
        if L != expected_len:
            raise ValueError(
                f"SFCDC expected Q/K length {expected_len} (= l_f*l_h*l_w + text_length), got {L}"
            )

        # Flatten Batch and Heads: [B*N, L, D]
        q_all = q.transpose(1, 2).reshape(-1, L, D).contiguous()
        k_all = k.transpose(1, 2).reshape(-1, L, D).contiguous()
        v_all = v.transpose(1, 2).reshape(-1, L, D).contiguous()
        B_all = q_all.size(0)

        # --- Split text / video tokens ---
        if T > 0:
            if self.text_location == 'begin':
                q_text, q_video = q_all[:, :T], q_all[:, T:]
                k_text, k_video = k_all[:, :T], k_all[:, T:]
                v_text, v_video = v_all[:, :T], v_all[:, T:]
            else:  # text_location == 'end'
                q_text, q_video = q_all[:, -T:], q_all[:, :-T]
                k_text, k_video = k_all[:, -T:], k_all[:, :-T]
                v_text, v_video = v_all[:, -T:], v_all[:, :-T]
            L_video = min(q_video.size(1), F_dim * H_dim * W_dim)
            q_video, k_video, v_video = q_video[:, :L_video], k_video[:, :L_video], v_video[:, :L_video]
            curr_L = T + L_video
        else:
            total_tokens = min(L, F_dim * H_dim * W_dim)
            if total_tokens < L:
                q_all, k_all, v_all = q_all[:, :total_tokens], k_all[:, :total_tokens], v_all[:, :total_tokens]
            q_video, k_video, v_video = q_all, k_all, v_all
            L_video = q_video.size(1)
            curr_L = L_video

        # --- Quantize video tokens only (text tokens bypass quantization) ---
        qv_int_raw, qv_s_inv = self._quantize_batch_blockwise(q_video, F_dim, H_dim, W_dim)
        kv_int_raw, kv_s_inv = self._quantize_batch_blockwise(k_video, F_dim, H_dim, W_dim)
        qv_int = qv_int_raw.to(torch.int8)
        kv_int = kv_int_raw.to(torch.int8)

        # ====== Phase 1: Clustering Warmup (video tokens only) ======
        if iter_idx < self.start_iter:
            chunk_size = 8
            for start_b in range(0, B_all, chunk_size):
                end_b = min(start_b + chunk_size, B_all)
                qi_chunk = qv_int[start_b:end_b].to(dtype)
                ki_chunk = kv_int[start_b:end_b].to(dtype)
                
                cq_init, ck_init = None, None
                try:
                    cq_list, ck_list = [], []
                    all_found = True
                    for i in range(start_b, end_b):
                        meta_key = (key, layer_idx, i % N)
                        if meta_key in self.offline_meta:
                            cq_list.append(self.offline_meta[meta_key]["centroids_q"])
                            ck_list.append(self.offline_meta[meta_key]["centroids_k"])
                        else:
                            all_found = False
                            break
                    if all_found:
                         cq_init = torch.stack(cq_list).to(device)
                         ck_init = torch.stack(ck_list).to(device)
                except Exception:
                    pass

                _, centroids_q, _ = batched_kmeans(qi_chunk, self.centers_q, init_centroids=cq_init, num_iters=1)
                _, centroids_k, _ = batched_kmeans(ki_chunk, self.centers_k, init_centroids=ck_init, num_iters=1)
                
                for i in range(start_b, end_b):
                    meta_key = (key, layer_idx, i % N)
                    self.offline_meta[meta_key] = {
                        "centroids_q": centroids_q[i - start_b].cpu(),
                        "centroids_k": centroids_k[i - start_b].cpu()
                    }
            return None

        # ====== Phase 2: Inference (Acceleration) ======
        out_all = torch.empty(B_all, curr_L, D, device=device, dtype=dtype)
        inv_sqrt_d = 1.0 / math.sqrt(D)
        chunk_size = 16
        
        for start_b in range(0, B_all, chunk_size):
            end_b = min(start_b + chunk_size, B_all)
            
            # Retrieve video centroids (Unscaled / quantized domain)
            try:
                sl_CQ = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_q"] for i in range(start_b, end_b)]).to(device)
                sl_CK = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_k"] for i in range(start_b, end_b)]).to(device)
            except KeyError:
                sl_CQ = torch.zeros(end_b - start_b, self.centers_q, D, device=device, dtype=dtype)
                sl_CK = torch.zeros(end_b - start_b, self.centers_k, D, device=device, dtype=dtype)
            
            sl_qi = qv_int[start_b:end_b]
            sl_ki = kv_int[start_b:end_b]
            sl_vi = v_video[start_b:end_b]
            sl_si_q = qv_s_inv[start_b:end_b]
            sl_si_k = kv_s_inv[start_b:end_b]

            if T > 0:
                # Text-aware compute core
                sl_qt = q_text[start_b:end_b]
                sl_kt = k_text[start_b:end_b]
                sl_vt = v_text[start_b:end_b]
                res, n_cq, n_ck = self._compute_core_with_text(
                    sl_qi, sl_ki, sl_vi,
                    sl_si_q, sl_si_k, sl_CQ, sl_CK,
                    sl_qt, sl_kt, sl_vt,
                    device, dtype, inv_sqrt_d, F_dim, H_dim, W_dim, L_video, T
                )
            else:
                # Original video-only compute core
                res, n_cq, n_ck = self._compute_core_triton_v4_2(
                    sl_qi, sl_ki, sl_vi,
                    sl_si_q, sl_si_k, sl_CQ, sl_CK,
                    device, dtype, inv_sqrt_d, F_dim, H_dim, W_dim, L_video
                )
            
            out_all[start_b:end_b].copy_(res, non_blocking=True)
            
            n_cq_cpu = n_cq.cpu()
            n_ck_cpu = n_ck.cpu()
            for i in range(start_b, end_b):
                meta_key = (key, layer_idx, i % N)
                self.offline_meta[meta_key]["centroids_q"] = n_cq_cpu[i - start_b]
                self.offline_meta[meta_key]["centroids_k"] = n_ck_cpu[i - start_b]

        # Reorder output: internal convention is [text, video].
        # If text_location='end', we need [video, text] to match input layout.
        if self.text_location == 'end' and T > 0:
            out_all = torch.cat([out_all[:, T:], out_all[:, :T]], dim=1)

        return out_all.reshape(B, N, curr_L, D).transpose(1, 2)

    def _compute_core_triton_v4_2(self, qi, ki, vi, si_q, si_k, CQ, CK, device, dtype, inv_sqrt_d, F_dim, H_dim, W_dim, L):
        B_slice = qi.size(0)
        qi_f, ki_f = qi.to(dtype), ki.to(dtype)
        
        # 1. Clustering Assignment (L2) - Online Re-assignment
        # Using Unscaled centroids vs Unscaled data -> Valid in Quant Domain
        c_LQ = self._reassign_clusters_online(qi_f, CQ, F_dim, H_dim, W_dim)
        c_LK = self._reassign_clusters_online(ki_f, CK, F_dim, H_dim, W_dim)
        
        # 2. Update Centroids (Mean) -> Result is Unscaled
        QC, _ = self._calculate_centroids(qi_f, c_LQ, self.centers_q, device, dtype)
        KC, KC_counts = self._calculate_centroids(ki_f, c_LK, self.centers_k, device, dtype)
        
        # Scale Handling (For Real Attention & Score Matrix Tiering)
        if si_q.size(1) < L: q_s_f = self._expand_scale(si_q, F_dim, H_dim, W_dim)[:, :L].to(dtype)
        else: q_s_f = si_q.to(dtype)

        if si_k.size(1) < L: k_s_f = self._expand_scale(si_k, F_dim, H_dim, W_dim)[:, :L].to(dtype)
        else: k_s_f = si_k.to(dtype)
             
        q_real = qi_f * q_s_f.unsqueeze(-1)
        k_real = ki_f * k_s_f.unsqueeze(-1)
        
        # --- NEW V4.3 LOGIC: Centroid Scaling ---
        # "Clustering center calculation S_c metrix needs consider quan factor"
        # We need "Real" Centroids for the Tier Table calculation to be accurate relative to real attention scores.
        # qc_scales: [B, Nc, 1]
        
        # Treat scales as a rank-3 tensor [B, L, 1] to reuse calculate_centroids
        # NOTE: q_s_f is float/bf16, dtype is passed explicitely
        qc_scales, _ = self._calculate_centroids(q_s_f.unsqueeze(-1), c_LQ, self.centers_q, device, dtype)
        kc_scales, _ = self._calculate_centroids(k_s_f.unsqueeze(-1), c_LK, self.centers_k, device, dtype)
        
        QC_real = QC * qc_scales
        KC_real = KC * kc_scales
        
        # 3. Compute Tier Table (B_slice, Nc, Nk)
        # Use Real Centroids for Score Matrix
        # [B, Nc, D] @ [B, D, Nk] -> [B, Nc, Nk]
        sc = torch.bmm(QC_real, KC_real.transpose(1, 2)) * inv_sqrt_d
        
        # Safety check for NaN/Inf
        if torch.isnan(sc).any() or torch.isinf(sc).any():
             sc = torch.nan_to_num(sc, nan=0.0, posinf=1e4, neginf=-1e4)

        # Tier selection
        sc_idx = torch.sort(sc, dim=-1, descending=True)[1]
        c0, c1, c2 = int(self.k0*self.centers_k), int(self.k1*self.centers_k), int(self.k2*self.centers_k)
        
        tier_table = torch.zeros_like(sc, dtype=torch.int32)
        # Optimize scatter usage
        tier_table.scatter_(2, sc_idx[:, :, :c0], 1)            # Tier 1: Exact
        tier_table.scatter_(2, sc_idx[:, :, c0:c0+c1], 2)       # Tier 2: Comp
        tier_table.scatter_(2, sc_idx[:, :, c0+c1:c0+c1+c2], 3) # Tier 3: Centroid
        
        # --- REORDERING & OPTIMIZATION ---
        # Sort indices based on cluster assignment
        idx_q_sorted = torch.argsort(c_LQ, dim=-1) # [B, L]
        idx_k_sorted = torch.argsort(c_LK, dim=-1) # [B, L]
        
        # Restore indices: argsort of argsort gives the position to place the sorted element back
        idx_q_restore = torch.argsort(idx_q_sorted, dim=-1)
        
        B_arange = torch.arange(B_slice, device=device).unsqueeze(1)
        
        # Gather Sorted Tensors 
        q_sorted = q_real[B_arange, idx_q_sorted]
        k_sorted = k_real[B_arange, idx_k_sorted]
        v_sorted = vi[B_arange, idx_k_sorted]
        lq_sorted = c_LQ[B_arange, idx_q_sorted]
        lk_sorted = c_LK[B_arange, idx_k_sorted]
        
        # Free large intermediate tensors no longer needed
        del q_real, k_real, qi_f, ki_f
        
        # Compute V_sums per Cluster (O=PV Optimization)
        # Use original-order vi and c_LK (both unsorted) for consistent pairing
        v_flat = vi.flatten(0, 1) # [B*L, D]
        # Global cluster IDs: batch_idx * num_clusters + cluster_idx
        lk_flat = (c_LK + (torch.arange(B_slice, device=device)*self.centers_k).unsqueeze(1)).flatten()
        v_sums_flat = torch.zeros(B_slice * self.centers_k, vi.shape[-1], device=device, dtype=dtype)
        v_sums_flat.index_add_(0, lk_flat, v_flat.to(dtype))
        v_sums = v_sums_flat.view(B_slice, self.centers_k, -1)
        
        # Prepare Cluster Offsets
        k_counts_hist = torch.zeros(B_slice, self.centers_k, device=device, dtype=torch.int32)
        k_counts_hist.scatter_add_(1, c_LK, torch.ones_like(c_LK, dtype=torch.int32))
        k_offsets = torch.cumsum(k_counts_hist, dim=1).to(torch.int32)
        k_starts = torch.cat([torch.zeros(B_slice, 1, device=device, dtype=torch.int32), k_offsets[:, :-1]], dim=1)
        k_ends = k_offsets
        
        # Pad Q/K/V/LQ to BLOCK_M=64 alignment
        BLOCK_M = 64
        L_actual = q_sorted.size(1)
        pad_len = (BLOCK_M - L_actual % BLOCK_M) % BLOCK_M
        if pad_len > 0:
            q_sorted = F.pad(q_sorted, (0, 0, 0, pad_len))
            k_sorted = F.pad(k_sorted, (0, 0, 0, pad_len))
            v_sorted = F.pad(v_sorted, (0, 0, 0, pad_len))
            lq_sorted = F.pad(lq_sorted, (0, pad_len), value=0)

        # Prepare Kernel Inputs
        q_in = q_sorted.unsqueeze(1).contiguous()
        k_in = k_sorted.unsqueeze(1).contiguous()
        v_in = v_sorted.unsqueeze(1).contiguous()
        
        # Tier 3 score = QC . KC. If QC/KC are Real, result is Real.
        # Tier 2 score = Q_real . KC.
        qc_in = QC_real.unsqueeze(1).contiguous()
        kc_in = KC_real.unsqueeze(1).contiguous()
        
        v_sums_in = v_sums.unsqueeze(1).contiguous()
        k_counts_in = k_counts_hist.unsqueeze(1).contiguous()
        
        lq_in = lq_sorted.unsqueeze(1).to(torch.int32).contiguous()
        k_starts_in = k_starts.unsqueeze(1).contiguous()
        k_ends_in = k_ends.unsqueeze(1).contiguous()
        tier_table_in = tier_table.unsqueeze(1).to(torch.int32).contiguous()
        
        # Call Optimus Prime (V4.2 Kernel)
        out_sorted = sfcdc_attention_v4_2(
            q_in, k_in, v_in,
            qc_in, kc_in,
            v_sums_in, k_counts_in,
            lq_in, k_starts_in, k_ends_in,
            tier_table_in,
            sm_scale=inv_sqrt_d
        )
        out_sorted = out_sorted.squeeze(1)[:, :L_actual]  # Remove padding
        out = out_sorted[B_arange, idx_q_restore]
        
        return out, QC.detach(), KC.detach()

    def _compute_core_with_text(self, qi_video, ki_video, vi_video,
                                si_q_video, si_k_video, CQ_video, CK_video,
                                q_text, k_text, v_text,
                                device, dtype, inv_sqrt_d,
                                F_dim, H_dim, W_dim, L_video, T):
        """
        SFCDC compute core with text token support (CogVideoX mode).
        
        Algorithm (conceptual):
          - Q_c = [Q_text_cluster, Q_video_centroids]   (centers_q + 1 clusters)
          - K_c = [K_video_centroids, K_text_cluster]    (centers_k + 1 clusters)  
          - Tier table: (centers_q + 1) x (centers_k + 1)
            - [video_q, video_k]: standard SFCDC tiering
            - [*, text_k_col]:    always tier 1 (exact)
            - [text_q_row, *]:    always tier 1 (exact)
          - Text tokens form single always-exact clusters, ensuring correct
            softmax normalization across both text and video keys.
        
        Args:
            qi_video, ki_video: [B, L_video, D] int8 quantized video tokens
            vi_video:           [B, L_video, D] original video values
            si_q_video, si_k_video: quantization scale inverses for video
            CQ_video, CK_video: [B, centers_q/k, D] video centroids (unscaled)
            q_text, k_text, v_text: [B, T, D] original text tokens
            L_video: number of video tokens
            T: number of text tokens
        """
        B_slice = qi_video.size(0)
        D = qi_video.size(2)
        Nc_q = self.centers_q + 1   # +1 for text Q cluster
        Nc_k = self.centers_k + 1   # +1 for text K cluster
        
        qi_f = qi_video.to(dtype)
        ki_f = ki_video.to(dtype)
        q_text_f = q_text.to(dtype)
        k_text_f = k_text.to(dtype)
        v_text_f = v_text.to(dtype)
        
        # ===== 1. Cluster Assignment (video tokens only) =====
        c_LQ_video = self._reassign_clusters_online(qi_f, CQ_video, F_dim, H_dim, W_dim)
        c_LK_video = self._reassign_clusters_online(ki_f, CK_video, F_dim, H_dim, W_dim)
        
        # ===== 2. Update Video Centroids (unscaled domain) =====
        QC_video_new, _ = self._calculate_centroids(qi_f, c_LQ_video, self.centers_q, device, dtype)
        KC_video_new, _ = self._calculate_centroids(ki_f, c_LK_video, self.centers_k, device, dtype)
        
        # ===== 3. Video Scale Handling =====
        if si_q_video.size(1) < L_video:
            q_s_f = self._expand_scale(si_q_video, F_dim, H_dim, W_dim)[:, :L_video].to(dtype)
        else:
            q_s_f = si_q_video.to(dtype)
        if si_k_video.size(1) < L_video:
            k_s_f = self._expand_scale(si_k_video, F_dim, H_dim, W_dim)[:, :L_video].to(dtype)
        else:
            k_s_f = si_k_video.to(dtype)
        
        # Dequantize video tokens to real values
        q_video_real = qi_f * q_s_f.unsqueeze(-1)
        k_video_real = ki_f * k_s_f.unsqueeze(-1)
        
        # ===== 4. Centroid Scaling (V4.3 logic) =====
        qc_scales, _ = self._calculate_centroids(q_s_f.unsqueeze(-1), c_LQ_video, self.centers_q, device, dtype)
        kc_scales, _ = self._calculate_centroids(k_s_f.unsqueeze(-1), c_LK_video, self.centers_k, device, dtype)
        QC_video_real = QC_video_new * qc_scales
        KC_video_real = KC_video_new * kc_scales
        
        # ===== 5. Text Centroids (already in real domain, no quantization) =====
        QC_text_real = q_text_f.mean(dim=1, keepdim=True)   # [B, 1, D]
        KC_text_real = k_text_f.mean(dim=1, keepdim=True)   # [B, 1, D]
        
        # ===== 6. Combined Centroids for Tier Scoring =====
        # Q centroids: [video_centroids, text_centroid] -> [B, Nc_q, D]
        # K centroids: [video_centroids, text_centroid] -> [B, Nc_k, D]
        QC_all_real = torch.cat([QC_video_real, QC_text_real], dim=1)
        KC_all_real = torch.cat([KC_video_real, KC_text_real], dim=1)
        
        # ===== 7. Score Matrix & Tier Table =====
        # Full score: [B, Nc_q, Nc_k]
        sc_full = torch.bmm(QC_all_real, KC_all_real.transpose(1, 2)) * inv_sqrt_d
        if torch.isnan(sc_full).any() or torch.isinf(sc_full).any():
            sc_full = torch.nan_to_num(sc_full, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Video-video sub-matrix tiering (standard SFCDC)
        sc_vv = sc_full[:, :self.centers_q, :self.centers_k]
        sc_vv_idx = torch.sort(sc_vv, dim=-1, descending=True)[1]
        c0 = int(self.k0 * self.centers_k)
        c1 = int(self.k1 * self.centers_k)
        c2 = int(self.k2 * self.centers_k)
        
        tier_table = torch.zeros(B_slice, Nc_q, Nc_k, device=device, dtype=torch.int32)
        # Video-video: standard tiering
        vv_tier = torch.zeros(B_slice, self.centers_q, self.centers_k, device=device, dtype=torch.int32)
        vv_tier.scatter_(2, sc_vv_idx[:, :, :c0], 1)
        vv_tier.scatter_(2, sc_vv_idx[:, :, c0:c0+c1], 2)
        vv_tier.scatter_(2, sc_vv_idx[:, :, c0+c1:c0+c1+c2], 3)
        tier_table[:, :self.centers_q, :self.centers_k] = vv_tier
        
        # Text K cluster column (last col): always tier 1 for all Q clusters
        tier_table[:, :, self.centers_k] = 1
        # Text Q cluster row (last row): always tier 1 for all K clusters
        tier_table[:, self.centers_q, :] = 1
        
        # ===== 8. Cluster Labels for ALL Tokens =====
        # Q: [text_tokens → cluster centers_q, video_tokens → 0..centers_q-1]
        c_LQ_text = torch.full((B_slice, T), self.centers_q, device=device, dtype=torch.int64)
        c_LQ = torch.cat([c_LQ_text, c_LQ_video], dim=1)  # [B, T+L_video]
        
        # K: [text_tokens → cluster centers_k, video_tokens → 0..centers_k-1]
        c_LK_text = torch.full((B_slice, T), self.centers_k, device=device, dtype=torch.int64)
        c_LK = torch.cat([c_LK_text, c_LK_video], dim=1)  # [B, T+L_video]
        
        # ===== 9. Combined Real Q/K/V =====
        q_real = torch.cat([q_text_f, q_video_real], dim=1)   # [B, T+L_video, D]
        k_real = torch.cat([k_text_f, k_video_real], dim=1)
        vi_all = torch.cat([v_text_f, vi_video.to(dtype)], dim=1)
        
        L_total = T + L_video
        
        # ===== 10. Sort by Cluster Assignment =====
        idx_q_sorted = torch.argsort(c_LQ, dim=-1)
        idx_k_sorted = torch.argsort(c_LK, dim=-1)
        idx_q_restore = torch.argsort(idx_q_sorted, dim=-1)
        
        B_arange = torch.arange(B_slice, device=device).unsqueeze(1)
        
        q_sorted = q_real[B_arange, idx_q_sorted]
        k_sorted = k_real[B_arange, idx_k_sorted]
        v_sorted = vi_all[B_arange, idx_k_sorted]
        lq_sorted = c_LQ[B_arange, idx_q_sorted]
        
        del q_real, k_real, qi_f, ki_f
        
        # ===== 11. V_sums per Cluster (Nc_k clusters including text) =====
        v_flat = vi_all.flatten(0, 1)
        lk_flat = (c_LK + (torch.arange(B_slice, device=device) * Nc_k).unsqueeze(1)).flatten()
        v_sums_flat = torch.zeros(B_slice * Nc_k, vi_all.shape[-1], device=device, dtype=dtype)
        v_sums_flat.index_add_(0, lk_flat, v_flat.to(dtype))
        v_sums = v_sums_flat.view(B_slice, Nc_k, -1)
        
        # ===== 12. K Cluster Offsets (Nc_k clusters) =====
        k_counts_hist = torch.zeros(B_slice, Nc_k, device=device, dtype=torch.int32)
        k_counts_hist.scatter_add_(1, c_LK, torch.ones_like(c_LK, dtype=torch.int32))
        k_offsets = torch.cumsum(k_counts_hist, dim=1).to(torch.int32)
        k_starts = torch.cat([torch.zeros(B_slice, 1, device=device, dtype=torch.int32), k_offsets[:, :-1]], dim=1)
        k_ends = k_offsets
        
        # ===== 13. Padding to BLOCK_M=64 =====
        BLOCK_M = 64
        L_actual = q_sorted.size(1)
        pad_len = (BLOCK_M - L_actual % BLOCK_M) % BLOCK_M
        if pad_len > 0:
            q_sorted = F.pad(q_sorted, (0, 0, 0, pad_len))
            k_sorted = F.pad(k_sorted, (0, 0, 0, pad_len))
            v_sorted = F.pad(v_sorted, (0, 0, 0, pad_len))
            lq_sorted = F.pad(lq_sorted, (0, pad_len), value=0)
        
        # ===== 14. Kernel Inputs =====
        q_in = q_sorted.unsqueeze(1).contiguous()
        k_in = k_sorted.unsqueeze(1).contiguous()
        v_in = v_sorted.unsqueeze(1).contiguous()
        qc_in = QC_all_real.unsqueeze(1).contiguous()
        kc_in = KC_all_real.unsqueeze(1).contiguous()
        v_sums_in = v_sums.unsqueeze(1).contiguous()
        k_counts_in = k_counts_hist.unsqueeze(1).contiguous()
        lq_in = lq_sorted.unsqueeze(1).to(torch.int32).contiguous()
        k_starts_in = k_starts.unsqueeze(1).contiguous()
        k_ends_in = k_ends.unsqueeze(1).contiguous()
        tier_table_in = tier_table.unsqueeze(1).to(torch.int32).contiguous()
        
        # ===== 15. Call Triton Kernel =====
        out_sorted = sfcdc_attention_v4_2(
            q_in, k_in, v_in,
            qc_in, kc_in,
            v_sums_in, k_counts_in,
            lq_in, k_starts_in, k_ends_in,
            tier_table_in,
            sm_scale=inv_sqrt_d
        )
        out_sorted = out_sorted.squeeze(1)[:, :L_actual]
        out = out_sorted[B_arange, idx_q_restore]
        
        # Return: combined output [B, T+L_video, D], video centroids (unscaled) for meta storage
        return out, QC_video_new.detach(), KC_video_new.detach()

    def _quantize_batch_blockwise(self, x_all: torch.Tensor, F_dim: int, H_dim: int, W_dim: int, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        B_all, L, D = x_all.shape
        HW = H_dim * W_dim
        n_g = (F_dim + self.group_size - 1) // self.group_size
        
        # Calculate padding needed to align with group size
        pad_needed = (n_g * self.group_size * HW) - L
        if pad_needed > 0:
            x_pad = F.pad(x_all, (0, 0, 0, pad_needed))
        else:
            x_pad = x_all
        x_g = x_pad.view(B_all, n_g, -1, D)
        
        # Compute max per group for scaling
        # Use simple max(abs) quantization
        m = x_g.abs().reshape(B_all, n_g, -1).max(dim=2, keepdim=True)[0]
        m = m.clamp(min=1e-6) # Prevent division by zero, slightly larger epsilon
        
        q_max = (1 << (bits - 1)) - 1
        scales = q_max / m 
        
        # Quantize
        x_int = (x_g * scales.unsqueeze(-1)).round().clamp(-q_max, q_max)
        s_inv = (1.0 / scales).squeeze(-1).to(x_all.dtype)
        
        return x_int.view(B_all, -1, D)[:, :L], s_inv

    def _expand_scale(self, s_inv: torch.Tensor, F_dim: int, H_dim: int, W_dim: int) -> torch.Tensor:
        HW = H_dim * W_dim
        n_g = s_inv.size(1)
        repeats = torch.full((n_g,), self.group_size, device=s_inv.device)
        return torch.repeat_interleave(s_inv, repeats * HW, dim=1)

    def _reassign_clusters_online(self, x_int: torch.Tensor, centroids: torch.Tensor, F_dim: int, H_dim: int, W_dim: int) -> torch.Tensor:
        """
        Re-assign tokens to clusters based on closest centroid distance.
        Optimized to respect Group-wise Cluster constraints (if any) or Global.
        Here we assume Global clustering split into groups for memory.
        """
        B_all, L, D = x_int.shape
        device = x_int.device
        
        # Heuristic to split computation if too many clusters to avoid OOM
        # But for correctness with the original logic:
        n_g = (F_dim + self.group_size - 1) // self.group_size
        num_centroids = centroids.size(1)
        
        k_base = num_centroids // n_g
        k_rem = num_centroids % n_g
        k_repeats = [k_base + 1 if i < k_rem else k_base for i in range(n_g)]
        max_k = max(k_repeats) if k_repeats else 0
        
        if max_k == 0:
            # Fallback if no centroids
            return torch.zeros(B_all, L, dtype=torch.int64, device=device)

        c_split = torch.split(centroids, k_repeats, dim=1)
        # Pad centroids to have same number per group for batched matmul
        c_g = torch.stack([F.pad(c, (0, 0, 0, max_k - c.size(1)), value=1e6) for c in c_split], dim=1)
        
        HW = H_dim * W_dim
        pad_size = (n_g * self.group_size * HW) - L
        if pad_size > 0:
            x_pad = F.pad(x_int, (0, 0, 0, pad_size))
        else:
            x_pad = x_int
            
        x_g = x_pad.view(B_all, n_g, -1, D)
        T_per_g = x_g.size(2)
        
        # Calculate distances
        x_flat = x_g.contiguous().view(-1, T_per_g, D)
        c_flat = c_g.contiguous().view(-1, max_k, D)
        
        # Ensure float for distance calculation
        if x_flat.dtype == torch.int8:
            x_flat = x_flat.float()
            
        dist = compiled_cdist(x_flat, c_flat) # [B*n_g, T, K]
        
        # Find min index
        l_local = torch.argmin(dist, dim=-1) # [B*n_g, T]
        l_local = l_local.view(B_all, n_g, T_per_g)
        
        # Convert local centroid index to global index
        l_offset = torch.tensor([0] + k_repeats[:-1], device=device).cumsum(0).view(1, n_g, 1)
        l_global = (l_local + l_offset).view(B_all, -1)[:, :L]
        
        # Safety clamp: prevent out-of-bounds when padded centroids are selected
        l_global = l_global.clamp(0, num_centroids - 1)
        
        return l_global

    def _calculate_centroids(self, x: torch.Tensor, labels: torch.Tensor, num_clusters: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.flatten(0, 1)
        # Offset labels by batch to do global aggregation
        l_flat = (labels + torch.arange(B, device=device).unsqueeze(1) * num_clusters).flatten()
        
        centroids_flat = torch.zeros(B * num_clusters, D, device=device, dtype=dtype)
        counts_flat = torch.zeros(B * num_clusters, device=device, dtype=dtype)
        
        # Use accumulating addition
        centroids_flat.index_add_(0, l_flat, x_flat.to(dtype))
        counts_flat.index_add_(0, l_flat, torch.ones_like(l_flat, dtype=dtype))
        
        centroids = centroids_flat.view(B, num_clusters, D)
        counts = counts_flat.view(B, num_clusters)
        
        # Avoid division by zero
        safe_counts = counts.clamp(min=1).unsqueeze(-1)
        
        return centroids / safe_counts, counts

    def _get_broadcast_scale(self, si, centers, L, dtype, device):
        return torch.ones(1, centers, 1, device=device, dtype=dtype)
