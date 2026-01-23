import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
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
        self.centroids_cache = {} 

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
        # [B, H, L, D] -> [B*H, L, D]
        q_flat = q_curr.flatten(0, 1)
        k_flat = k_curr.flatten(0, 1)

        # 3. K-Means
        cache_id = f"{key}_q_{layer_idx}"
        init_c_q = self.centroids_cache.get(cache_id)
        
        if init_c_q is not None and init_c_q.shape[0] != B * H:
            init_c_q = None

        # 聚类 (已在 utils 中高度优化)
        q_labels, q_centroids, _ = batched_kmeans(q_flat, self.num_clusters_q, init_centroids=init_c_q)
        k_labels, k_centroids, k_counts = batched_kmeans(k_flat, self.num_clusters_k)
        
        self.centroids_cache[cache_id] = q_centroids.detach()

        # 4. 计算连接图
        connectivity_map = compute_connectivity_map(q_centroids, k_centroids, k_counts, top_p=self.top_p)

        # Reshape [B*H, ...] -> [B, H, ...]
        q_labels = q_labels.view(B, H, L)
        k_labels = k_labels.view(B, H, L)
        connectivity_map = connectivity_map.view(B, H, self.num_clusters_q, self.num_clusters_k)

        # 5. 定义 FlexAttention 的 Mask Mod
        def sap_score_mod(score, b, h, q_idx, kv_idx):
            c_q = q_labels[b, h, q_idx]
            c_k = k_labels[b, h, kv_idx]
            is_connected = connectivity_map[b, h, c_q, c_k]
            return torch.where(is_connected, 0, float("-inf")) + score

        # 6. 执行 Attention (带 Block Mask 加速)
        try:
            # 关键优化: 创建 Block Mask
            # 这里的 _compile=True 会缓存 Mask 的生成逻辑
            block_mask = create_block_mask(
                sap_score_mod, 
                B, H, L, L, 
                device=q_curr.device, 
                _compile=True 
            )
            
            # 使用 block_mask 可以跳过大量计算
            out = flex_attention(q_curr, k_curr, v_curr, score_mod=sap_score_mod, block_mask=block_mask)
            return out.transpose(1, 2)
        except Exception as e:
            # print(f"SVG Error: {e}, Fallback to Dense")
            return None