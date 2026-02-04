# SFCDC version: 4.0
# Fix Q/K quantization to block, dist_metric to L2, centroid_mode to mean
# Simplified logic for Tier 2 and Tier 3
import torch
import torch.nn.functional as F
import math
import threading

class SFCDC_Simulator:
    def __init__(self, enabled=False, clusbegin_iter=11, start_iter=12, end_iter=46, centers_q=256, centers_k=256, group_size=4, k0=0.25, k1=0.25, k2=0.50, l_f=16, l_h=16, l_w=16, q_quant_type='block', k_quant_type='token', dist_metric='L2', centroid_mode='mean', diff_mode='queue'):
        self.enabled = enabled
        
        if start_iter == 0:
            raise ValueError("Start_iter cannot be 0 in SFCDC v4")
        
        self.start_iter = start_iter
        self.clusbegin_iter = start_iter - 1
        self.end_iter = end_iter
        self.centers_q = centers_q
        self.centers_k = centers_k
        self.group_size = group_size  
        self.k0 = k0                  
        self.k1 = k1                  
        self.k2 = k2                  
        
        # Fixed parameters for v4
        self.q_quant_type = 'block'
        self.k_quant_type = 'block'
        self.dist_metric = 'L2'           # Fixed: L2
        self.centroid_mode = 'mean'       # Fixed: mean
        self.diff_mode = 'query'          # Fixed: query
        
        self.l_f = l_f
        self.l_h = l_h
        self.l_w = l_w

        self.offline_meta = {}

    def analyze(self, q, k, v, layer_idx, iter_idx, key):
        if not self.enabled or iter_idx < self.clusbegin_iter or iter_idx > self.end_iter: 
            return None

        B, L, N, D = q.shape
        F_dim, H_dim, W_dim = self.l_f, self.l_h, self.l_w
        total_tokens = min(L, F_dim * H_dim * W_dim)
        device, dtype = q.device, q.dtype

        # 整理维度
        q_all = q.transpose(1, 2).reshape(-1, L, D)[:, :total_tokens]
        k_all = k.transpose(1, 2).reshape(-1, L, D)[:, :total_tokens]
        v_all = v.transpose(1, 2).reshape(-1, L, D)[:, :total_tokens]
        B_all = q_all.size(0)
        curr_L = q_all.size(1)

        # 1. 显存优化：强制 Block 量化
        q_int_raw, q_s_inv = self._quantize_batch_blockwise(q_all, F_dim, H_dim, W_dim)
        k_int_raw, k_s_inv = self._quantize_batch_blockwise(k_all, F_dim, H_dim, W_dim)
            
        q_int, k_int = q_int_raw.to(torch.int8), k_int_raw.to(torch.int8)

        num_gpus = torch.cuda.device_count()
        gpu_splits = []
        heads_per_gpu = (B_all + num_gpus - 1) // num_gpus
        for i in range(num_gpus):
            start, end = i * heads_per_gpu, min((i+1) * heads_per_gpu, B_all)
            if start < end:
                gpu_splits.append((start, end, torch.device(f"cuda:{i}")))

        if iter_idx < self.start_iter:
            def train_worker(start, end, target_dev):
                for i in range(start, end):
                    meta_key = (key, layer_idx, i % N)
                    prev_res = self.offline_meta.get(meta_key, None)
                    
                    curr_q_s = q_s_inv[i].to(target_dev)
                    curr_k_s = k_s_inv[i].to(target_dev)
                    
                    # Block 量化展开
                    sq_i = self._expand_scale(curr_q_s.unsqueeze(0), F_dim, H_dim, W_dim)[0, :curr_L].unsqueeze(-1)
                    sk_i = self._expand_scale(curr_k_s.unsqueeze(0), F_dim, H_dim, W_dim)[0, :curr_L].unsqueeze(-1)

                    qi = (q_int[i].to(target_dev, dtype=dtype) * sq_i)
                    ki = (k_int[i].to(target_dev, dtype=dtype) * sk_i)
                    
                    res = self._run_offline_clustering(qi, ki, F_dim, H_dim, W_dim, prev_meta=prev_res, device=target_dev)
                    res["centroids_q"] = res["centroids_q"].to("cpu")
                    res["centroids_k"] = res["centroids_k"].to("cpu")
                    self.offline_meta[meta_key] = res

            threads = [threading.Thread(target=train_worker, args=s) for s in gpu_splits]
            for t in threads: t.start()
            for t in threads: t.join()
            return None

        # 阶段 C: 加速推理
        out_all = torch.empty_like(q_all)
        inv_sqrt_d = 1.0 / math.sqrt(D)

        def inference_worker(start_b, end_b, target_dev):
            sl_CQ = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_q"] for i in range(start_b, end_b)]).to(target_dev)
            sl_CK = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_k"] for i in range(start_b, end_b)]).to(target_dev)
            sl_qi = q_int[start_b:end_b].to(target_dev)
            sl_ki = k_int[start_b:end_b].to(target_dev)
            sl_si_q = q_s_inv[start_b:end_b].to(target_dev)
            sl_si_k = k_s_inv[start_b:end_b].to(target_dev)
            sl_v = v_all[start_b:end_b].to(target_dev, dtype=dtype)

            chunk_size = 6
            for s in range(0, end_b - start_b, chunk_size):
                e = min(s + chunk_size, end_b - start_b)
                res, n_cq, n_ck = self._compute_core(
                    sl_qi[s:e], sl_ki[s:e], sl_v[s:e],
                    sl_si_q[s:e], sl_si_k[s:e], sl_CQ[s:e], sl_CK[s:e],
                    target_dev, dtype, inv_sqrt_d, F_dim, H_dim, W_dim, curr_L
                )
                out_all[start_b + s : start_b + e].copy_(res, non_blocking=True)
                for i in range(s, e):
                    idx = start_b + i
                    self.offline_meta[(key, layer_idx, idx % N)]["centroids_q"] = n_cq[i-s].to("cpu")
                    self.offline_meta[(key, layer_idx, idx % N)]["centroids_k"] = n_ck[i-s].to("cpu")

        threads = [threading.Thread(target=inference_worker, args=s) for s in gpu_splits]
        for t in threads: t.start()
        for t in threads: t.join()

        return out_all.reshape(B, N, curr_L, D).transpose(1, 2)

    def _compute_core(self, qi, ki, vi, si_q, si_k, CQ, CK, device, dtype, inv_sqrt_d, F_dim, H_dim, W_dim, L):
        B_slice = qi.size(0)
        qi_f, ki_f = qi.to(dtype), ki.to(dtype)
        
        # 1. 聚类分配 (Fixed: L2)
        c_LQ = self._reassign_clusters_online(qi_f, CQ, F_dim, H_dim, W_dim)
        c_LK = self._reassign_clusters_online(ki_f, CK, F_dim, H_dim, W_dim)
        
        # 2. 中心计算 (Fixed: Mean)
        QC, _ = self._calculate_centroids(qi_f, c_LQ, self.centers_q, device, dtype)
        KC, KC_counts = self._calculate_centroids(ki_f, c_LK, self.centers_k, device, dtype)
        
        # Scale 处理 (Fixed: Block)
        if si_q.size(1) < L:
             q_s_f = self._expand_scale(si_q, F_dim, H_dim, W_dim)[:, :L].to(dtype)
        else:
             q_s_f = si_q.to(dtype)

        if si_k.size(1) < L:
             k_s_f = self._expand_scale(si_k, F_dim, H_dim, W_dim)[:, :L].to(dtype)
        else:
             k_s_f = si_k.to(dtype)

        # 准备近似 Score 用于排序 (v3 Logic kept for metric consistency)
        kc_ss = torch.zeros((B_slice, self.centers_k, 1), device=device, dtype=dtype)
        kc_ss.scatter_add_(1, c_LK.unsqueeze(-1), k_s_f.unsqueeze(-1))
        ks_c = (kc_ss / KC_counts.clamp(min=1))

        n_g = si_q.size(1)
        k_base, k_rem = self.centers_q // n_g, self.centers_q % n_g
        k_repeats = torch.full((n_g,), k_base, device=device)
        k_repeats[:k_rem] += 1
        qs_c = torch.repeat_interleave(si_q.to(dtype), k_repeats, dim=1).unsqueeze(-1)
        
        # (B, CQ, CK)
        sc = torch.bmm((QC * qs_c), (KC * ks_c).transpose(1, 2)) * inv_sqrt_d
        sc_idx = torch.sort(sc, dim=-1, descending=True)[1]
        
        c0, c1, c2 = int(self.k0*self.centers_k), int(self.k1*self.centers_k), int(self.k2*self.centers_k)
        
        s = torch.full((B_slice, L, L), -10000.0, device=device, dtype=dtype)

        # --- Tier 1 (Exact) ---
        # 保持对应位置矩阵乘法
        m_e = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, :c0], True)
        m_e_exp = m_e.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers_k)).gather(2, c_LK.unsqueeze(1).expand(-1, L, -1))
        del m_e 
        res_e = torch.bmm(qi_f, ki_f.transpose(1, 2))
        b, r, c = m_e_exp.nonzero(as_tuple=True)
        s[b, r, c] = (res_e[b, r, c] * q_s_f[b, r] * k_s_f[b, c] * inv_sqrt_d).to(dtype)
        del m_e_exp, res_e

        # --- Tier 2 (Compensation - Fixed: Query Mode Optimized) ---
        # 使用 Q * K_c 替代原有的 Q_c K_c + (Q-Q_c) K_c
        m_d = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, c0:c0+c1], True)
        m_d_exp = m_d.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers_k)).gather(2, c_LK.unsqueeze(1).expand(-1, L, -1))
        del m_d

        # 1. 计算 Q * KC^T -> (B, L, CK)
        res_d_base = torch.bmm(qi_f, KC.transpose(1, 2))
        
        # 2. 只需要 gather K 的方向即可 (因为 Q 维已经是 L 了)
        # res_d_base 本身包含了 i 到所有 K cluster 的距离
        # 如果 (i, j) 被选中，我们取 res_d_base[i, c_LK[j]]
        b, r, c = m_d_exp.nonzero(as_tuple=True)
        k_cluster_ids = c_LK[b, c]
        
        s[b, r, c] = (res_d_base[b, r, k_cluster_ids] * q_s_f[b, r] * k_s_f[b, c] * inv_sqrt_d).to(dtype)
        del m_d_exp, res_d_base

        # --- Tier 3 (Centroid) ---
        # 直接使用已有的 Q_c K_c 结果 (sc) 进行广播
        m_c = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, c0+c1:c0+c1+c2], True)
        m_c_exp = m_c.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers_k)).gather(2, c_LK.unsqueeze(1).expand(-1, L, -1))
        del m_c
        b, r, c = m_c_exp.nonzero(as_tuple=True)
        s[b, r, c] = sc[b, c_LQ[b, r], c_LK[b, c]].to(dtype)
        del m_c_exp

        res_out = torch.bmm(F.softmax(s, dim=-1), vi)
        return res_out, QC.detach(), KC.detach()

    def _quantize_batch_blockwise(self, x_all, F_dim, H_dim, W_dim, bits=8):
        B_all, L, D = x_all.shape
        HW = H_dim * W_dim
        n_g = (F_dim + self.group_size - 1) // self.group_size
        pad_size = n_g * self.group_size * HW
        x_pad = F.pad(x_all, (0, 0, 0, pad_size - L))
        x_g = x_pad.view(B_all, n_g, -1, D)
        m = x_g.abs().reshape(B_all, n_g, -1).max(dim=2, keepdim=True)[0].clamp(min=1e-8)
        q_max = (1 << (bits - 1)) - 1
        scales = q_max / m 
        x_int = (x_g * scales.unsqueeze(-1)).round().clamp(-q_max, q_max)
        s_inv = (1.0 / scales).squeeze(-1).to(x_all.dtype)
        return x_int.view(B_all, -1, D)[:, :L], s_inv

    def _expand_scale(self, s_inv, F_dim, H_dim, W_dim):
        HW = H_dim * W_dim
        n_g = s_inv.size(1)
        repeats = torch.full((n_g,), self.group_size, device=s_inv.device)
        return torch.repeat_interleave(s_inv, repeats * HW, dim=1)

    def _reassign_clusters_online(self, x_int, centroids, F_dim, H_dim, W_dim):
        B_all, L, D = x_int.shape
        device = x_int.device
        n_g = (F_dim + self.group_size - 1) // self.group_size
        
        k_base, k_rem = centroids.size(1) // n_g, centroids.size(1) % n_g
        k_repeats = [k_base + 1 if i < k_rem else k_base for i in range(n_g)]
        max_k = max(k_repeats)
        
        c_split = torch.split(centroids, k_repeats, dim=1)
        c_g = torch.stack([F.pad(c, (0, 0, 0, max_k - c.size(1)), value=1e6) for c in c_split], dim=1)
        
        HW = H_dim * W_dim
        pad_size = n_g * self.group_size * HW
        x_g = F.pad(x_int, (0, 0, 0, pad_size - L)).view(B_all, n_g, -1, D)
        
        T_per_g = x_g.size(2)
        # Fixed: L2 metric (p=2) impl implicitly by cdist default
        dist = torch.cdist(x_g.view(-1, T_per_g, D), c_g.view(-1, max_k, D))
        l = torch.argmin(dist.view(B_all, n_g, T_per_g, max_k), dim=-1)
        
        offsets = torch.zeros(n_g, device=device, dtype=torch.long)
        curr_off = 0
        for i, r in enumerate(k_repeats):
            offsets[i] = curr_off
            curr_off += r
            
        l_global = (l + offsets.view(1, n_g, 1)).view(B_all, -1)
        return l_global[:, :L]

    def _run_offline_clustering(self, q, k, F_dim, H_dim, W_dim, prev_meta=None, device=None):
        L, D = q.shape[0], q.shape[1]
        dtype = q.dtype
        n_g = (F_dim + self.group_size - 1) // self.group_size
        
        def get_kpg_list(total_centers):
            k_base, k_rem = total_centers // n_g, total_centers % n_g
            return [k_base + 1 if i < k_rem else k_base for i in range(n_g)]
        
        kpg_q = get_kpg_list(self.centers_q)
        kpg_k = get_kpg_list(self.centers_k)
        
        def clus(data, p_all, kpg_list):
            ls, cs, off = [], [], 0
            tpg = self.group_size * H_dim * W_dim
            for g_idx in range(n_g):
                g = data[g_idx*tpg : min((g_idx+1)*tpg, L)]
                if g.size(0) == 0: continue
                curr_k = kpg_list[g_idx]
                ic = p_all[off : off + curr_k].to(device) if p_all is not None else None
                _, l, c = self._kmeans_basic(g, curr_k, initial_centroids=ic, device=device)
                ls.append(l + off); cs.append(c); off += curr_k
            return torch.cat(ls), torch.cat(cs)
            
        p_q = prev_meta["centroids_q"] if prev_meta else None
        p_k = prev_meta["centroids_k"] if prev_meta else None
        lq, cq = clus(q, p_q, kpg_q)
        lk, ck = clus(k, p_k, kpg_k)
        return {"labels_q": lq, "centroids_q": cq, "labels_k": lk, "centroids_k": ck}

    def _calculate_centroids(self, data, labels, num_centers, device, dtype):
        """Fixed to Mean mode"""
        B_sl, L, D = data.shape
        counts = torch.zeros((B_sl, num_centers, 1), device=device, dtype=dtype)
        counts.scatter_add_(1, labels.unsqueeze(-1), torch.ones_like(labels, dtype=dtype).unsqueeze(-1))
        m = torch.zeros((B_sl, num_centers, D), device=device, dtype=dtype)
        m.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, D), data)
        return (m / counts.clamp(min=1)).round(), counts

    def _kmeans_basic(self, x, k, n_iter=5, initial_centroids=None, device=None):
        dtype, N, D = x.dtype, x.shape[0], x.shape[1]
        
        if initial_centroids is not None and initial_centroids.size(0) == k:
            c = initial_centroids.clone().to(dtype)
        else:
            # Initialization
            c = torch.empty((k, D), device=device, dtype=dtype)
            curr_idx = torch.randint(N, (1,), device=device).item()
            c[0] = x[curr_idx]
            dist_sq = torch.cdist(x, c[0:1]).squeeze(1) ** 2
            
            for i in range(1, k):
                if dist_sq.sum() > 1e-6:
                    curr_idx = torch.multinomial(dist_sq.clamp(min=0), 1).item()
                else:
                    curr_idx = torch.randint(N, (1,), device=device).item()
                c[i] = x[curr_idx]
                new_dist_sq = torch.cdist(x, c[i:i+1]).squeeze(1) ** 2
                dist_sq = torch.minimum(dist_sq, new_dist_sq)
            c = c.to(dtype)
        
        for _ in range(n_iter):
            # Fixed: L2
            dist = torch.cdist(x, c) ** 2
            l = torch.argmin(dist, dim=1)
            new_c, _ = self._calculate_centroids(x.unsqueeze(0), l.unsqueeze(0), k, device, dtype)
            c = new_c.squeeze(0)
        return None, l, c.to(dtype)
