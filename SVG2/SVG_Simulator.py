import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from .svg_utils import batched_kmeans, compute_connectivity_map

class SVG_Simulator:
    def __init__(self, enabled=False, start_iter=0, start_layer=0, sparsity=0.9, num_clusters_q=500, num_clusters_k=100, cluster_update_interval=1):
        """
        cluster_update_interval: 每隔多少个 Iteration 更新一次聚类。
        """
        self.enabled = enabled
        self.start_iter = start_iter
        self.start_layer = start_layer
        self.top_p = sparsity
        self.num_clusters_q = num_clusters_q
        self.num_clusters_k = num_clusters_k
        self.update_interval = cluster_update_interval
        
        # 缓存跨帧初始化质心 (Temporal Consistency)
        self.centroids_cache = {} 
        
        # 缓存当前 Iteration 范围内的状态 (Interval Clustering)
        # key: "{cond_key}_{layer_idx}" -> value: { 'q_labels': ..., 'map': ... }
        self.state_cache = {}

    def run(self, q, k, v, layer_idx, iter_idx, grid_sizes, key="default"):
        """
        Args:
            q, k, v: [B, L, H, D]
        """
        if not self.enabled: return None
        if iter_idx < self.start_iter or layer_idx < self.start_layer: return None
        if key is None: key = "default"

        q_curr = q.transpose(1, 2)
        k_curr = k.transpose(1, 2)
        v_curr = v.transpose(1, 2)
        
        B, H, L, D = q_curr.shape
        
        # 状态缓存 Key
        state_key = f"{key}_{layer_idx}"
        
        # 判断是否需要更新聚类 (Check Update Interval)
        # 如果是 start_iter 或者是 interval 的倍数，则更新
        should_update = (iter_idx - self.start_iter) % self.update_interval == 0
        
        # 尝试获取缓存状态
        cached_state = self.state_cache.get(state_key)
        
        # 如果需要更新，或者没有缓存，则执行聚类
        if should_update or cached_state is None:
            # -------------------------------------------------
            # 执行聚类与计算 Mask (Heavy Path)
            # -------------------------------------------------
            
            # [B, H, L, D] -> [B*H, L, D]
            q_flat = q_curr.flatten(0, 1)
            k_flat = k_curr.flatten(0, 1)

            # 获取上一帧质心初始化
            init_c_q = self.centroids_cache.get(state_key) # 复用 key
            if init_c_q is not None and init_c_q.shape[0] != B * H: init_c_q = None

            q_labels, q_centroids, _ = batched_kmeans(q_flat, self.num_clusters_q, init_centroids=init_c_q)
            k_labels, k_centroids, k_counts = batched_kmeans(k_flat, self.num_clusters_k)
            
            # 保存用于下一帧初始化的质心
            self.centroids_cache[state_key] = q_centroids.detach()

            connectivity_map = compute_connectivity_map(q_centroids, k_centroids, k_counts, top_p=self.top_p)

            # Reshape
            q_labels = q_labels.view(B, H, L)
            k_labels = k_labels.view(B, H, L)
            connectivity_map = connectivity_map.view(B, H, self.num_clusters_q, self.num_clusters_k)
            
            # 更新状态缓存
            cached_state = {
                'q_labels': q_labels,
                'k_labels': k_labels,
                'map': connectivity_map
            }
            self.state_cache[state_key] = cached_state
        
        # -------------------------------------------------
        # 使用当前/缓存的状态执行 Attention
        # -------------------------------------------------
        q_labels = cached_state['q_labels']
        k_labels = cached_state['k_labels']
        connectivity_map = cached_state['map']

        def sap_score_mod(score, b, h, q_idx, kv_idx):
            c_q = q_labels[b, h, q_idx]
            c_k = k_labels[b, h, kv_idx]
            is_connected = connectivity_map[b, h, c_q, c_k]
            return torch.where(is_connected, 0, float("-inf")) + score

        try:
            # 注意: 缓存 Block Mask 本身比较复杂，因为 L 可能变化(虽然通常不变)
            # 但 create_block_mask 带有 _compile=True 已经涵盖了大部分优化
            block_mask = create_block_mask(
                sap_score_mod, 
                B, H, L, L, 
                device=q_curr.device, 
                _compile=True 
            )
            out = flex_attention(q_curr, k_curr, v_curr, score_mod=sap_score_mod, block_mask=block_mask)
            return out.transpose(1, 2)
        except Exception:
            return None