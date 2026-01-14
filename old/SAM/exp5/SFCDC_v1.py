# version 1.0
# 单GPU
import torch
import torch.nn.functional as F
import math

class SFCDC_Simulator:
    def __init__(self, enabled=False, start_iter=2, end_iter=45, centers=256, group_size=4, k0=0.02, k1=0.08, k2=0.15, l_f=16, l_h=16, l_w=16):
        self.enabled = enabled
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.centers = centers        
        self.group_size = group_size  
        self.k0 = k0                  # Tier 1: 全值计算比例 (Exact)
        self.k1 = k1                  # Tier 2: 差分补偿比例 (Diff)
        self.k2 = k2                  # Tier 3: 仅质心比例 (Centroid)
        
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

        # 整理维度并准备量化
        q_all = q.transpose(1, 2).reshape(-1, L, D)[:, :total_tokens]
        k_all = k.transpose(1, 2).reshape(-1, L, D)[:, :total_tokens]
        v_all = v.transpose(1, 2).reshape(-1, L, D)[:, :total_tokens]
        B_all = q_all.size(0)
        curr_L = q_all.size(1)

        # 1. 差异化量化：Q (Block-wise), K (Token-wise)
        q_int, q_s_inv = self._quantize_batch_blockwise(q_all, F_dim, H_dim, W_dim)
        k_int, k_s_inv = self._quantize_tokenwise(k_all)
        
        # 阶段 A: 背景训练期
        if iter_idx < self.start_iter:
            sq_all = self._expand_scale(q_s_inv, F_dim, H_dim, W_dim)[:, :curr_L].unsqueeze(-1)
            sk_all = k_s_inv.unsqueeze(-1)
            
            for i in range(B_all):
                meta_key = (key, layer_idx, i % N)
                self.offline_meta[meta_key] = self._run_offline_clustering(
                    q_int[i].to(dtype) * sq_all[i], k_int[i].to(dtype) * sk_all[i],
                    F_dim, H_dim, W_dim, prev_meta=self.offline_meta.get(meta_key)
                )
            return None

        if iter_idx > self.end_iter: return None

        # 阶段 C: 加速推理期 (先更新质心数值，再计算加速)
        C_Q_prev = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_q"] for i in range(B_all)])
        C_K_prev = torch.stack([self.offline_meta[(key, layer_idx, i % N)]["centroids_k"] for i in range(B_all)])

        out_all = torch.empty_like(q_all)
        inv_sqrt_d = 1.0 / math.sqrt(D)
        chunk_size = 4

        for s_b in range(0, B_all, chunk_size):
            e_b = min(s_b + chunk_size, B_all)
            cur_B = e_b - s_b
            
            c_qi_int, c_ki_int = q_int[s_b:e_b], k_int[s_b:e_b]
            c_si_q, c_si_k = q_s_inv[s_b:e_b], k_s_inv[s_b:e_b]
            
            # A. 在线重分配 (仅返回 L1 距离映射标签)
            c_LQ = self._reassign_clusters_online(c_qi_int, C_Q_prev[s_b:e_b], F_dim, H_dim, W_dim)
            c_LK = self._reassign_clusters_online(c_ki_int, C_K_prev[s_b:e_b], F_dim, H_dim, W_dim)
            
            # B. 质心更新 (全量 Token 取均值)
            def compute_cluster_means(data, labels):
                B, L, D = data.shape
                means = torch.zeros((B, self.centers, D), device=device, dtype=dtype)
                counts = torch.zeros((B, self.centers, 1), device=device, dtype=dtype)
                means.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, D), data.to(dtype))
                counts.scatter_add_(1, labels.unsqueeze(-1), torch.ones_like(labels).unsqueeze(-1).to(dtype))
                return (means / counts.clamp(min=1)).round(), counts

            QC_I, _ = compute_cluster_means(c_qi_int, c_LQ)
            KC_I, KC_counts = compute_cluster_means(c_ki_int, c_LK)
            
            # C. 簇级幅度修正 (KC_scale)
            # 计算分配到每个簇的 Token Scale 的平均值
            kc_scales_sum = torch.zeros((cur_B, self.centers, 1), device=device, dtype=dtype)
            kc_scales_sum.scatter_add_(1, c_LK.unsqueeze(-1), c_si_k.unsqueeze(-1).to(dtype))
            ks_c = (kc_scales_sum / KC_counts.clamp(min=1))

            n_g = c_si_q.size(1)
            k_base, k_rem = self.centers // n_g, self.centers % n_g
            k_repeats = torch.full((n_g,), k_base, device=device)
            k_repeats[:k_rem] += 1
            
            qs_c = torch.repeat_interleave(c_si_q.to(dtype), k_repeats, dim=1).unsqueeze(-1)
            
            # D. 加速推理
            # 显式强制质心得分矩阵为 dtype
            sc = torch.bmm((QC_I * qs_c).to(dtype), (KC_I * ks_c).transpose(1, 2).to(dtype)) * inv_sqrt_d
            sc_idx = torch.sort(sc, dim=-1, descending=True)[1]
            
            c0, c1, c2 = int(self.k0*self.centers), int(self.k1*self.centers), int(self.k2*self.centers)
            q_s_full = self._expand_scale(c_si_q, F_dim, H_dim, W_dim)[:, :curr_L].to(dtype)
            k_s_full = c_si_k.to(dtype)
            
            s = torch.full((cur_B, curr_L, curr_L), -10000.0, device=device, dtype=dtype)

            # Tier 1: Exact
            m_e_base = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, :c0], True)
            m_e = m_e_base.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1, curr_L, -1))
            del m_e_base
            
            res_e = torch.bmm(c_qi_int.to(dtype), c_ki_int.to(dtype).transpose(1, 2))
            b_idx, r_idx, c_idx = m_e.nonzero(as_tuple=True)
            res_e_m = res_e[b_idx, r_idx, c_idx]
            scale_m = q_s_full[b_idx, r_idx] * k_s_full[b_idx, c_idx] * inv_sqrt_d
            s[b_idx, r_idx, c_idx] = (res_e_m * scale_m).to(dtype)
            del m_e, res_e, b_idx, r_idx, c_idx, res_e_m, scale_m

            # Tier 3: Centroid (直接从 sc 获取，避免实例化 L*L 的 sc_exp)
            m_c_base = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, c0+c1:c0+c1+c2], True)
            m_c = m_c_base.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1, curr_L, -1))
            del m_c_base
            
            b_idx, r_idx, c_idx = m_c.nonzero(as_tuple=True)
            q_c_ids = c_LQ[b_idx, r_idx]
            k_c_ids = c_LK[b_idx, c_idx]
            s[b_idx, r_idx, c_idx] = sc[b_idx, q_c_ids, k_c_ids].to(dtype)
            del m_c, b_idx, r_idx, c_idx, q_c_ids, k_c_ids

            # Tier 2: Diff
            m_d_base = torch.zeros_like(sc, dtype=torch.bool).scatter_(2, sc_idx[:, :, c0:c0+c1], True)
            m_d = m_d_base.gather(1, c_LQ.unsqueeze(-1).expand(-1, -1, self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1, curr_L, -1))
            del m_d_base

            qc_t = torch.gather(QC_I, 1, c_LQ.unsqueeze(-1).expand(-1, -1, D))
            kc_t = torch.gather(KC_I, 1, c_LK.unsqueeze(-1).expand(-1, -1, D))
            q_diff = (c_qi_int.to(dtype) - qc_t)
            k_diff = (c_ki_int.to(dtype) - kc_t)
            del qc_t, kc_t

            res_d = torch.bmm(QC_I, KC_I.transpose(1, 2)).gather(1, c_LQ.unsqueeze(-1).expand(-1,-1,self.centers)).gather(2, c_LK.unsqueeze(1).expand(-1,curr_L,-1))
            t2 = torch.bmm(q_diff, KC_I.transpose(1, 2)).gather(2, c_LK.unsqueeze(1).expand(-1,curr_L,-1))
            res_d.add_(t2); del t2
            t3 = torch.bmm(QC_I, k_diff.transpose(1, 2)).gather(1, c_LQ.unsqueeze(-1).expand(-1,-1,curr_L))
            res_d.add_(t3); del t3, q_diff, k_diff
            
            b_idx, r_idx, c_idx = m_d.nonzero(as_tuple=True)
            res_d_m = res_d[b_idx, r_idx, c_idx]
            scale_m = q_s_full[b_idx, r_idx] * k_s_full[b_idx, c_idx] * inv_sqrt_d
            s[b_idx, r_idx, c_idx] = (res_d_m * scale_m).to(dtype)
            del m_d, res_d, b_idx, r_idx, c_idx, res_d_m, scale_m
            
            out_all[s_b:e_b] = torch.bmm(F.softmax(s, dim=-1), v_all[s_b:e_b].to(dtype))
            
            # 更新状态池
            for i in range(s_b, e_b):
                self.offline_meta[(key, layer_idx, i % N)]["centroids_q"] = QC_I[i-s_b].detach()
                self.offline_meta[(key, layer_idx, i % N)]["centroids_k"] = KC_I[i-s_b].detach()

            del s, sc, sc_idx, QC_I, KC_I, c_LQ, c_LK

        return out_all.reshape(B, N, curr_L, D).transpose(1, 2)

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


    def _run_offline_clustering(self, q, k, F_dim, H_dim, W_dim, prev_meta=None):
        D, L = q.shape[-1], q.shape[0]
        dtype = q.dtype
        n_g = (F_dim + self.group_size - 1) // self.group_size
        
        # 分配质心数
        k_base, k_rem = self.centers // n_g, self.centers % n_g
        kpg_list = [k_base + 1 if i < k_rem else k_base for i in range(n_g)]
        
        def clus(data, p_all):
            df = data.view(L, D)
            ls, cs, off = [], [], 0
            tpg = self.group_size * H_dim * W_dim
            for g_idx in range(n_g):
                g = df[g_idx*tpg : min((g_idx+1)*tpg, L)]
                if g.size(0) == 0: continue
                # 对应组的质心数
                curr_k = kpg_list[g_idx]
                ic = p_all[off : off + curr_k].to(dtype) if p_all is not None else None
                _, l, c = self._kmeans_basic(g, curr_k, initial_centroids=ic)
                ls.append(l + off); cs.append(c.to(dtype)); off += curr_k
            return torch.cat(ls), torch.cat(cs)
            
        lq, cq = clus(q, prev_meta["centroids_q"] if prev_meta else None)
        lk, ck = clus(k, prev_meta["centroids_k"] if prev_meta else None)
        return {"labels_q": lq, "centroids_q": cq, "labels_k": lk, "centroids_k": ck}

    def _kmeans_basic(self, x, k, n_iter=5, initial_centroids=None):
        device, dtype, N, D = x.device, x.dtype, x.shape[0], x.shape[1]
        c = initial_centroids.clone() if (initial_centroids is not None and initial_centroids.size(0) == k) else x[torch.randperm(N, device=device)[:k]]
        c = c.to(dtype)
        for _ in range(n_iter):
            l = torch.argmin(torch.cdist(x, c, p=2)**2, dim=1)
            nc, cnt = torch.zeros_like(c), torch.zeros(k, 1, device=device, dtype=dtype)
            nc.index_add_(0, l, x); cnt.index_add_(0, l, torch.ones(N, 1, device=device, dtype=dtype))
            c = nc / (cnt + 1e-6)
        return None, l, c.to(dtype)