import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from .svg_utils import batched_kmeans, compute_connectivity_map

class SVG_Simulator:
    def __init__(self, enabled=False, start_iter=0, start_layer=0, sparsity=0.9, num_clusters_q=500, num_clusters_k=100):
        """
        sparsity: 这里对应 SVG 论文中的 top-k 累积概率 P (默认0.9)
        num_clusters: Q 的聚类中心数目 (num_clusters_q)
        num_clusters_k: K 的聚类中心数目。如果为 None，则默认等于 num_clusters。
        """
        self.enabled = enabled
        self.start_iter = start_iter
        self.start_layer = start_layer
        self.top_p = sparsity
        self.num_clusters_q = num_clusters_q
        self.num_clusters_k = num_clusters_k
        
        # 缓存上一帧的质心，用于初始化下一帧 (Temporal Consistency)
        # key: "{cond_key}_q_{layer_idx}" -> value: centroids [B, K, D]
        self.centroids_cache = {} 
        
        if enabled:
            print(f"SVG Simulator (SAP) Init: P={self.top_p}, K_q={self.num_clusters_q}, K_k={self.num_clusters_k}")

    def run(self, q, k, v, layer_idx, iter_idx, grid_sizes, key="default"):
        """
        Args:
            q, k, v: [B, L, H, D]
            key: 用于区分 cond/uncond 的缓存 Key
        """
        if not self.enabled: return None
        if iter_idx < self.start_iter or layer_idx < self.start_layer: return None
        if key is None: key = "default"

        # 1. 转换 Layout [B, L, H, D] -> [B, H, L, D]
        q_curr = q.transpose(1, 2)
        k_curr = k.transpose(1, 2)
        v_curr = v.transpose(1, 2)
        
        B, H, L, D = q_curr.shape

        # 2. 准备聚类数据 (Per-Head Clustering)
        # 每个 Head 独立聚类。
        # [B, H, L, D] -> [B*H, L, D]
        q_flat = q_curr.flatten(0, 1)
        k_flat = k_curr.flatten(0, 1)

        # 3. K-Means
        # 尝试获取上一帧的质心作为初始化
        cache_id = f"{key}_q_{layer_idx}"
        init_c_q = self.centroids_cache.get(cache_id)
        
        # 检查缓存有效性 (Batch * Head 数量是否匹配)
        if init_c_q is not None and init_c_q.shape[0] != B * H:
            init_c_q = None

        # Q 聚类 (使用初始化)
        # 注意：现在 batched_kmeans 返回 3 个值
        q_labels, q_centroids, _ = batched_kmeans(q_flat, self.num_clusters_q, init_centroids=init_c_q)
        
        # K 聚类 - 获取 k_counts
        # 输出 Centroids: [B*H, K_k, D], Counts: [B*H, K_k]
        k_labels, k_centroids, k_counts = batched_kmeans(k_flat, self.num_clusters_k)
        
        # 更新缓存
        self.centroids_cache[cache_id] = q_centroids.detach()

        # 4. 计算连接图 [B*H, K_q, K_k]
        # 传入 k_counts 以进行 Weighted Softmax
        connectivity_map = compute_connectivity_map(q_centroids, k_centroids, k_counts, top_p=self.top_p)

        # Reshape 回 [B, H, ...] 以便在 Attention 中索引
        # q_labels: [B*H, L] -> [B, H, L]
        q_labels = q_labels.view(B, H, L)
        k_labels = k_labels.view(B, H, L)
        # connectivity_map: [B*H, K_q, K_k] -> [B, H, K_q, K_k]
        connectivity_map = connectivity_map.view(B, H, self.num_clusters_q, self.num_clusters_k)

        # 5. 定义 FlexAttention 的 Mask Mod
        # 这里的闭包捕获了 labels 和 map
        def sap_score_mod(score, b, h, q_idx, kv_idx):
            # score: [B, H, M, N] 当前计算出的注意力分数(未softmax)
            
            # 查找 Cluster ID (Per-Head)
            c_q = q_labels[b, h, q_idx]
            c_k = k_labels[b, h, kv_idx]
            
            # 查表 (Per-Head)
            is_connected = connectivity_map[b, h, c_q, c_k]
            
            # Mask: connected -> Keep; Not -> Mask
            return torch.where(is_connected, 0, float("-inf")) + score

        # 6. 执行 Attention
        try:
            # score_mod 修改 Attention 分数，实现 mask 效果
            out = flex_attention(q_curr, k_curr, v_curr, score_mod=sap_score_mod)
            return out.transpose(1, 2)
        except Exception as e:
            # print(f"SVG Error: {e}, Fallback to Dense")
            return None