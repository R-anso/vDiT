import torch
import triton
import triton.language as tl

@triton.jit
def _sfcdc_attn_fwd_kernel_v4_2(
    Q, K, V,                # [B, H, L, D] (Sorted)
    sample_Qc, Kc,          # [B, H, Nc, D], [B, H, Nk, D]
    V_sums,                 # [B, H, Nk, D] (Aggregated V for Tier 2/3)
    K_counts,               # [B, H, Nk] (Aggregated Count for Tier 2/3 Denom)
    L_q,                    # [B, H, L] (Query Cluster Labels for Tier Lookup)
    K_starts, K_ends,       # [B, H, Nk] (Start/End indices in K array for Tier 1)
    Tier_Table,             # [B, H, Nc, Nk]
    Out,                    # [B, H, L, D]
    # Strides
    stride_b, stride_h, stride_n, stride_d,
    stride_c_b, stride_c_h, stride_c_n, stride_c_d,
    stride_vsum_b, stride_vsum_h, stride_vsum_n, stride_vsum_d,
    stride_kcnt_b, stride_kcnt_h, stride_kcnt_n,
    stride_l_b, stride_l_h, stride_l_n,
    stride_ks_b, stride_ks_h, stride_ks_n,
    stride_t_b, stride_t_h, stride_t_x, stride_t_y,
    # Constants
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    Nc: tl.constexpr, Nk: tl.constexpr
):
    # Grid: (L/BLOCK_M, H, B)
    off_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    
    # Base Pointers
    # Q Ptr: For loading the current block of Queries
    Q_ptr = Q + (off_b * stride_b + off_h * stride_h + off_m * stride_n)
    
    # L_q Ptr: To find out which cluster these queries belong to
    Lq_ptr = L_q + (off_b * stride_l_b + off_h * stride_l_h + off_m * stride_l_n)
    
    # V_sum, K_counts, Kc, TierTable, K_starts, K_ends depend on Batch/Head
    # Common offsets
    off_bh_c = (off_b * stride_c_b + off_h * stride_c_h)
    off_bh_vsum = (off_b * stride_vsum_b + off_h * stride_vsum_h)
    off_bh_kcnt = (off_b * stride_kcnt_b + off_h * stride_kcnt_h)
    off_bh_ks = (off_b * stride_ks_b + off_h * stride_ks_h)
    off_bh_t = (off_b * stride_t_b + off_h * stride_t_h)
    
    # Load Q Block (Assumed to be sorted, so mostly same cluster, but maybe mixed)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_n + offs_d[None, :])
    q = tl.load(q_ptrs) # [BLOCK_M, D]
    
    lq = tl.load(Lq_ptr + offs_m) # [BLOCK_M]
    
    # Load Qc for this block (Approximation: Use Gather or if Q sorted, 1-2 loads)
    # Using Gather for generality
    sqc_ptrs = sample_Qc + off_bh_c + (lq[:, None] * stride_c_n + offs_d[None, :])
    qc = tl.load(sqc_ptrs) # [BLOCK_M, D]
    
    # Initializes Accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Iterate over Key Clusters (Nk)
    # This is the V4.2 Core Loop: Iterate Clusters, not tokens!
    for k_idx in range(Nk):
        
        # 1. Determine Tier for each Query in Block against Cluster k_idx
        # Lq = [M], k_idx is scalar
        # Look up Tier Table
        tier_ptr_offsets = lq * stride_t_x + k_idx * stride_t_y
        tiers = tl.load(Tier_Table + off_bh_t + tier_ptr_offsets) # [BLOCK_M]
        
        # Load Cluster Meta: Kc, V_sum, K_count for cluster k_idx
        kc_ptr = Kc + off_bh_c + (k_idx * stride_c_n + offs_d[None, :]) # [1, D]
        kc_vec = tl.load(kc_ptr) # [1, D] (or broadcasted to M if needed?)
        # We need Kc for dot product.
        
        vsum_ptr = V_sums + off_bh_vsum + (k_idx * stride_vsum_n + offs_d[None, :])
        vsum_vec = tl.load(vsum_ptr) # [1, D]
        
        kcnt_val = tl.load(K_counts + off_bh_kcnt + k_idx * stride_kcnt_n) # Scalar
        
        # Compute "Approximate" Scores (Tier 2 & 3)
        # Tier 3: Qc . Kc_T
        s_cent = tl.sum(qc * kc_vec, axis=1) # [BLOCK_M] 
        
        # Tier 2: Q . Kc_T
        s_comp = tl.sum(q * kc_vec, axis=1) # [BLOCK_M]

        is_exact = (tiers == 1)
        is_approx = (tiers != 1)
        
        if tl.sum(is_approx, axis=0) > 0: # Optimization: skip if all exact
            # Choose score
            s_approx = tl.where(tiers == 3, s_cent, s_comp) # [BLOCK_M]
            s_approx = s_approx * sm_scale
            
            # Online Softmax Update (Scalar broadcast style)
            m_prev = m_i
            m_new = tl.maximum(m_prev, s_approx)
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_prev - m_new))
            
            # New contribution P_new = exp(s_approx - m_new)
            # If s_approx = -inf, p_new = 0.
            p_new = tl.exp(s_approx - m_new)
            
            # Update Denom: l_i * alpha + p_new * count
            l_new = l_i * alpha + p_new * kcnt_val
            
            # Update Num: acc * alpha + p_new * V_sum
            # Broadcast p_new [M, 1] * vsum_vec [1, D]
            acc = acc * alpha[:, None] + p_new[:, None] * vsum_vec
            
            m_i = m_new
            l_i = l_new
        
        # 2. Handle Tier 1 (Exact)
        # Get Cluster Bounds
        k_start = tl.load(K_starts + off_bh_ks + k_idx * stride_ks_n)
        k_end = tl.load(K_ends + off_bh_ks + k_idx * stride_ks_n)
        need_exact = (tl.sum(is_exact, axis=0) > 0)
        
        if need_exact and (k_end > k_start):
            k_base_ptr = K + (off_b * stride_b + off_h * stride_h)
            v_base_ptr = V + (off_b * stride_b + off_h * stride_h)
            
            # Loop over K chunks in this cluster
            # This handles "Continuous Memory" optimization of SVG2
            for ks in range(k_start, k_end, BLOCK_N):
                # How many tokens valid?
                curr_block_k = min(BLOCK_N, k_end - ks)
                
                # Setup pointers for this sub-block
                offs_n_sub = ks + tl.arange(0, BLOCK_N)
                mask_k = (offs_n_sub < k_end) # Mask for boundary
                
                k_ptrs_Real = k_base_ptr + (offs_n_sub[None, :] * stride_n + offs_d[:, None])
                v_ptrs_Real = v_base_ptr + (offs_n_sub[:, None] * stride_n + offs_d[None, :])
                
                k_val = tl.load(k_ptrs_Real, mask=mask_k[None, :], other=0.0)
                v_val = tl.load(v_ptrs_Real, mask=mask_k[:, None], other=0.0)
                
                # Standard Attn
                s_real = tl.dot(q, k_val) * sm_scale
                
                s_real = tl.where(is_exact[:, None] & mask_k[None, :], s_real, float("-inf"))
                
                # Online Softmax Update
                m_prev = m_i
                m_new = tl.maximum(m_prev, tl.max(s_real, 1))
                
                # [Fix] Prevent NaN check for Tier 1
                alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_prev - m_new))
                
                p_real = tl.exp(s_real - m_new[:, None])
                
                acc = acc * alpha[:, None] + tl.dot(p_real.to(tl.float16), v_val.to(tl.float16))
                l_i = l_i * alpha + tl.sum(p_real, 1)
                m_i = m_new

    # Store
    l_i = tl.where(l_i == 0.0, 1.0e-6, l_i)
    acc = acc / l_i[:, None]
    out_ptrs = Out + (off_b * stride_b + off_h * stride_h + off_m * stride_n + offs_m[:, None] * stride_n + offs_d[None, :])
    tl.store(out_ptrs, acc)

def sfcdc_attention_v4_2(q, k, v, qc, kc, v_sums, k_counts, lq, k_starts, k_ends, tier_table, sm_scale=1.0):
    B, H, L, D = q.shape
    Nc = qc.shape[2]
    Nk = kc.shape[2]
    
    out = torch.empty_like(q)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(L, BLOCK_M), H, B)
    
    _sfcdc_attn_fwd_kernel_v4_2[grid](
        q, k, v,
        qc, kc,
        v_sums, k_counts,
        lq, k_starts, k_ends,
        tier_table,
        out,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qc.stride(0), qc.stride(1), qc.stride(2), qc.stride(3),
        v_sums.stride(0), v_sums.stride(1), v_sums.stride(2), v_sums.stride(3),
        k_counts.stride(0), k_counts.stride(1), k_counts.stride(2),
        lq.stride(0), lq.stride(1), lq.stride(2),
        k_starts.stride(0), k_starts.stride(1), k_starts.stride(2),
        tier_table.stride(0), tier_table.stride(1), tier_table.stride(2), tier_table.stride(3),
        # Consts
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        Nc=Nc, Nk=Nk,
        num_warps=4, num_stages=2
    )
    return out
