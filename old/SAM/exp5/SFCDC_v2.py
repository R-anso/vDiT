import torch
import torch.nn.functional as F
import math
import threading

class SFCDC_Simulator:
    def __init__(self, enabled=False, start_iter=2, end_iter=45, centers=256, group_size=4, k0=0.02, k1=0.08, k2=0.15, l_f=16, l_h=16, l_w=16):
        self.enabled = enabled
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.centers = centers        
        self.group_size = group_size  
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        
        self.l_f = l_f                
        self.l_h = l_h                
        self.l_w = l_w                
        self.offline_meta = {}

    def analyze(self, q, k, v, layer_idx, iter_idx, key):
        if not self.enabled: return None

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

        # 1. 显存优化：使用 int8 存储量化后的数据 [Save 75% VRAM for these buffers]
        q_int, q_s_inv = self._quantize_batch_blockwise(q_all, F_dim, H_dim, W_dim)
        k_int, k_s_inv = self._quantize_tokenwise(k_all)
        q_int, k_int = q_int.to(torch.int8), k_int.to(torch.int8)

        num_gpus = torch.cuda.device_count()
        gpu_splits = []
        heads_per_gpu = (B_all + num_gpus - 1) // num_gpus
        for i in range(num_gpus):
            start, end = i * heads_per_gpu, min((i+1) * heads_per_gpu, B_all)
            if start < end:
                gpu_splits.append((start, end, torch.device(f"cuda:{i}")))

        if iter_idx < self.start_iter:
            sq_all = self._expand_scale(q_s_inv, F_dim, H_dim, W_dim)[:, :curr_L].unsqueeze(-1)
            sk_all = k_s_inv.unsqueeze(-1)
            
            def train_worker(start, end, target_dev):
                for i in range(start, end):
                    # 转换回 dtype 进行计算
                    qi = (q_int[i].to(target_dev, dtype=dtype) * sq_all[i].to(target_dev))
                    ki = (k_int[i].to(target_dev, dtype=dtype) * sk_all[i].to(target_dev))
                    res = self._run_offline_clustering(qi, ki, F_dim, H_dim, W_dim, device=target_dev)
                    res["centroids_q"] = res["centroids_q"].to("cpu")
                    res["centroids_k"] = res["centroids_k"].to("cpu")
                    self.offline_meta[(key, layer_idx, i % N)] = res

            threads = [threading.Thread(target=train_worker, args=s) for s in gpu_splits]
            for t in threads: t.start()
            for t in threads: t.join()
            return None

        if iter_idx > self.end_iter: return None

        # 阶段 C: 加速推理
        out_all = torch.empty_like(q_all)
        inv_sqrt_d = 1.0 / math.sqrt(D)

        def inference_worker(start_b, end_b, target_dev):
            # [显存优化] 直接从 CPU 字典加载质心，不经过 cuda:0 全量堆叠
            sl_CQ = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_q"] for i in range(start_b, end_b)]).to(target_dev)
            sl_CK = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_k"] for i in range(start_b, end_b)]).to(target_dev)
            
            sl_qi = q_int[start_b:end_b].to(target_dev)
            sl_ki = k_int[start_b:end_b].to(target_dev)
            sl_si_q = q_s_inv[start_b:end_b].to(target_dev)
            sl_si_k = k_s_inv[start_b:end_b].to(target_dev)
            sl_v = v_all[start_b:end_b].to(target_dev, dtype=dtype)

            chunk_size = 4
            for s in range(0, end_b - start_b, chunk_size):
                e = min(s + chunk_size, end_b - start_b)
                # 执行核心计算 (此时 qi/ki 仍是 int8)
                res, n_cq, n_ck = self._compute_core(
                    sl_qi[s:e], sl_ki[s:e], sl_v[s:e],
                    sl_si_q[s:e], sl_si_k[s:e], sl_CQ[s:e], sl_CK[s:e],
                    target_dev, dtype, inv_sqrt_d, F_dim, H_dim, W_dim, curr_L
                )
                out_all[start_b + s : start_b + e].copy_(res, non_blocking=True)
                # 质心存回 CPU
                for i in range(s, e):
                    idx = start_b + i
                    self.offline_meta[(key, layer_idx, idx % N)]["centroids_q"] = n_cq[i-s].to("cpu")
                    self.offline_meta[(key, layer_idx, idx % N)]["centroids_k"] = n_ck[i-s].to("cpu")

        threads = [threading.Thread(target=inference_worker, args=s) for s in gpu_splits]
        for t in threads: t.start()
        for t in threads: t.join()

        return out_all.reshape(B, N, curr_L, D).transpose(1, 2)

    def _compute_core(self, qi, ki, vi, si_q, si_k, CQ, CK, device, dtype, inv_sqrt_d, F_dim, H_dim, W_dim, L):
        """完全约束在单个 device 上的核心计算函数"""
        B_slice = qi.size(0)
        D_dim = qi.size(2)
        qi_f, ki_f = qi.to(dtype), ki.to(dtype)
        
        c_LQ = self._reassign_clusters_online(qi_f, CQ, F_dim, H_dim, W_dim)
        c_LK = self._reassign_clusters_online(ki_f, CK, F_dim, H_dim, W_dim)
        
        # B. 质心更新
        QC, _ = self._local_means(qi_f, c_LQ, device, dtype)
        KC, KC_counts = self._local_means(ki_f, c_LK, device, dtype)
        
        # C. 簇级幅度修正
        kc_ss = torch.zeros((B_slice, self.centers, 1), device=device, dtype=dtype)
        kc_ss.scatter_add_(1, c_LK.unsqueeze(-1), si_k.unsqueeze(-1).to(dtype))
        ks_c = (kc_ss / KC_counts.clamp(min=1))

        n_g = si_q.size(1)
        k_base, k_rem = self.centers // n_g, self.centers % n_g
        k_repeats = torch.full((n_g,), k_base, device=device)
        k_repeats[:k_rem] += 1
        qs_c = torch.repeat_interleave(si_q.to(dtype), k_repeats, dim=1).unsqueeze(-1)
        
        # D. Tiered Attention
        sc = torch.bmm((QC * qs_c), (KC * ks_c).transpose(1, 2)) * inv_sqrt_d
        sc_idx = torch.sort(sc, dim=-1, descending=True)[1]
        c0, c1, c2 = int(self.k0*self.centers), int(self.k1*self.centers), int(self.k2*self.centers)
        
        s = torch.full((qi.size(0), L, L), -10000.0, device=device, dtype=dtype)
        q_s_f = self._expand_scale(si_q, F_dim, H_dim, W_dim)[:, :L].to(dtype)
        k_s_f = si_k.to(dtype)

        # Tier 1 (Exact)
        m_e = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, :int(self.k0*self.centers)], True)
        m_e_exp = m_e.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1, L, -1))
        del m_e 
        res_e = torch.bmm(qi_f, ki_f.transpose(1, 2))
        b, r, c = m_e_exp.nonzero(as_tuple=True)
        s[b, r, c] = (res_e[b, r, c] * q_s_f[b, r] * k_s_f[b, c] * inv_sqrt_d).to(dtype)
        del m_e_exp, res_e

        # Tier 3 (Centroid)
        m_c = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, c0+c1:c0+c1+c2], True)
        m_c_exp = m_c.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1, L, -1))
        del m_c
        b, r, c = m_c_exp.nonzero(as_tuple=True)
        s[b, r, c] = sc[b, c_LQ[b, r], c_LK[b, c]].to(dtype)
        del m_c_exp

        # Tier 2 (Diff)
        m_d = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, c0:c0+c1], True)
        m_d_exp = m_d.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1, L, -1))
        del m_d
        qc_t = torch.gather(QC, 1, c_LQ.unsqueeze(-1).expand(-1, -1, D_dim))
        kc_t = torch.gather(KC, 1, c_LK.unsqueeze(-1).expand(-1, -1, D_dim))
        q_diff, k_diff = (qi.to(dtype) - qc_t), (ki.to(dtype) - kc_t)
        res_d = torch.bmm(QC, KC.transpose(1, 2)).gather(1, c_LQ.unsqueeze(-1).expand(-1,-1,self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1,L,-1))
        res_d.add_(torch.bmm(q_diff, KC.transpose(1, 2)).gather(2, c_LK.unsqueeze(1).expand(-1,L,-1)))
        res_d.add_(torch.bmm(QC, k_diff.transpose(1, 2)).gather(1, c_LQ.unsqueeze(-1).expand(-1,-1,L)))
        b, r, c = m_d_exp.nonzero(as_tuple=True)
        s[b, r, c] = (res_d[b, r, c] * q_s_f[b, r] * k_s_f[b, c] * inv_sqrt_d).to(dtype)
        del m_d_exp, res_d, qc_t, kc_t, q_diff, k_diff

        res_out = torch.bmm(F.softmax(s, dim=-1), vi)
        return res_out, QC.detach(), KC.detach()

    def _quantize_tokenwise(self, x, bits=8):
        B_all, L, D = x.shape
        q_max = (1 << (bits - 1)) - 1
        scales = q_max / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        x_int = (x * scales).round().clamp(-q_max, q_max)
        s_inv = (1.0 / scales).squeeze(-1)
        return x_int, s_inv

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
        """返回 L1 距离最近的质心 ID，支持非均匀分配"""
        B_all, L, D = x_int.shape
        device = x_int.device
        HW = H_dim * W_dim
        n_g = (F_dim + self.group_size - 1) // self.group_size
        
        # 计算每组质心分布
        k_base, k_rem = centroids.size(1) // n_g, centroids.size(1) % n_g
        k_repeats = [k_base + 1 if i < k_rem else k_base for i in range(n_g)]
        max_k = max(k_repeats)
        
        # 1. 划分并填充质心到统一形状
        c_split = torch.split(centroids, k_repeats, dim=1)
        c_g = torch.stack([F.pad(c, (0, 0, 0, max_k - c.size(1)), value=1e6) for c in c_split], dim=1)
        
        # 2. 准备数据组
        pad_size = n_g * self.group_size * HW
        x_g = F.pad(x_int, (0, 0, 0, pad_size - L)).view(B_all, n_g, -1, D)
        
        # 3. 计算距离并获取索引
        dist = torch.norm(x_g.unsqueeze(3) - c_g.unsqueeze(2), p=1, dim=-1)
        l = torch.argmin(dist, dim=-1) # 每个 group 内部的索引
        
        # 4. 计算全局偏移并返回单张量 (不再返回元组)
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
        k_base, k_rem = self.centers // n_g, self.centers % n_g
        kpg_list = [k_base + 1 if i < k_rem else k_base for i in range(n_g)]
        
        def clus(data, p_all):
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
        lq, cq = clus(q, p_q)
        lk, ck = clus(k, p_k)
        return {"labels_q": lq, "centroids_q": cq, "labels_k": lk, "centroids_k": ck}

    def _kmeans_basic(self, x, k, n_iter=5, initial_centroids=None, device=None):
        dtype, N, D = x.dtype, x.shape[0], x.shape[1]
        if initial_centroids is not None and initial_centroids.size(0) == k:
            c = initial_centroids.clone().to(dtype)
        else:
            c = x[torch.randperm(N, device=device)[:k]].to(dtype)
        
        for _ in range(n_iter):
            # 严格在传入的 device 上计算
            dist = torch.cdist(x, c, p=2)**2
            l = torch.argmin(dist, dim=1)
            nc = torch.zeros_like(c)
            cnt = torch.zeros((k, 1), device=device, dtype=dtype)
            nc.index_add_(0, l, x)
            cnt.index_add_(0, l, torch.ones((N, 1), device=device, dtype=dtype))
            c = nc / (cnt + 1e-6)
        return None, l, c.to(dtype)

    def _local_means(self, data, labels, device, dtype):
        m = torch.zeros((data.size(0), self.centers, data.size(2)), device=device, dtype=dtype)
        c = torch.zeros((data.size(0), self.centers, 1), device=device, dtype=dtype)
        m.scatter_add_(1, labels.unsqueeze(-1).expand(-1,-1,data.size(2)), data)
        c.scatter_add_(1, labels.unsqueeze(-1), torch.ones_like(labels).unsqueeze(-1).to(dtype))
        return (m / c.clamp(min=1)).round(), c