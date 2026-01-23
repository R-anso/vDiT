import torch
import torch.nn.functional as F

# 尝试启用 torch.compile 加速
try:
    compile_mode = "reduce-overhead"
    @torch.compile(mode=compile_mode)
    def compiled_cdist(x, y):
        return torch.cdist(x, y, p=2)
except:
    def compiled_cdist(x, y):
        return torch.cdist(x, y, p=2)

def batched_kmeans(x, num_clusters, num_iters=5, init_centroids=None):
    """
    针对 Batch 数据的 K-Means 实现 (高度优化版)。
    优化点:
    1. 移除 Python 循环初始化，改用向量化 Gather。
    2. 移除 one_hot + bmm 更新，改用 index_add_ (大幅省显存和带宽)。
    """
    B, N, D = x.shape
    device = x.device
    
    if init_centroids is None:
        # 向量化随机初始化 (比循环快得多)
        # 生成随机索引 [B, K]
        rand_cols = torch.randint(0, N, (B, num_clusters), device=device)
        # 生成 Batch 偏移 [B, K] -> 0,0,0... N,N,N... 2N,2N,2N...
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * N
        flat_indices = (rand_cols + batch_offsets).flatten()
        
        # [B*N, D] -> gather -> [B*K, D] -> view
        centroids = x.flatten(0, 1)[flat_indices].view(B, num_clusters, D)
    else:
        centroids = init_centroids

    # 预先生成 batch 偏移用于 labels 展平
    # labels_offset: [B, 1] -> broadcast to [B, N]
    labels_offset = (torch.arange(B, device=device) * num_clusters).unsqueeze(1)
    
    for _ in range(num_iters):
        # 1. 计算距离
        dists = compiled_cdist(x, centroids) # [B, N, K]
        
        # 2. E步: 分配标签
        labels = torch.argmin(dists, dim=-1) # [B, N]

        # 3. M步: 更新质心 (使用 index_add_ 替代 one_hot matmul)
        # 展平以便通过 scatter add 操作
        x_flat = x.flatten(0, 1) # [B*N, D]
        labels_flat = (labels + labels_offset).flatten() # [B*N] 全局聚类ID
        
        # 准备累加容器 [B*K, D]
        new_centroids_flat = torch.zeros(B * num_clusters, D, device=device, dtype=x.dtype)
        counts_flat = torch.zeros(B * num_clusters, device=device, dtype=x.dtype)
        
        # 核心加速: 消除巨大的中间 one_hot 张量
        new_centroids_flat.index_add_(0, labels_flat, x_flat)
        
        # 计算计数 (对于 float 类型可以直接用 index_add 加上 1.0)
        ones = torch.ones_like(labels_flat, dtype=x.dtype)
        counts_flat.index_add_(0, labels_flat, ones)
        
        # 还原形状
        new_centroids = new_centroids_flat.view(B, num_clusters, D)
        counts = counts_flat.view(B, num_clusters)
        
        # 避免除以0
        centroids = new_centroids / (counts.unsqueeze(-1) + 1e-6)
        
    return labels, centroids, counts

# compute_connectivity_map 通常只运行一次，开销较小，但也可以加 compile
@torch.compile
def compute_connectivity_map(q_centroids, k_centroids, k_counts, top_p=0.9):
    # ...existing code...
    # 1. 簇间相似度 (Dot Product)
    scale = q_centroids.shape[-1] ** -0.5
    scores = torch.bmm(q_centroids, k_centroids.transpose(1, 2)) * scale # [B, K_q, K_k]
    
    # 2. Weighted Softmax (复现官方逻辑)
    weights = k_counts.unsqueeze(1).float() # [B, 1, K_k]
    
    # 为了数值稳定性: exp(x - max) * w / sum(...)
    max_score = scores.max(dim=-1, keepdim=True)[0]
    weighted_exp = torch.exp(scores - max_score) * weights
    # 避免除以0
    sum_weighted_exp = torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    probs = weighted_exp / sum_weighted_exp
    
    # 3. Top-P Selection (按行)
    # 降序排列
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # SVG逻辑: shift trick
    shifted_cumsum = torch.cat(
        [torch.zeros_like(cumsum_probs[:, :, :1]), cumsum_probs[:, :, :-1]], 
        dim=-1
    )
    selected_mask_sorted = shifted_cumsum < top_p
    
    # 4. 还原顺序
    final_mask = torch.zeros_like(probs, dtype=torch.bool)
    final_mask.scatter_(2, sorted_indices, selected_mask_sorted)
    
    return final_mask