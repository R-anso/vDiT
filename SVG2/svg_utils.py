import torch
import torch.nn.functional as F

def batched_kmeans(x, num_clusters, num_iters=5, init_centroids=None):
    """
    针对 Batch 数据的 K-Means 实现。
    Returns: 
        labels: [B, N]
        centroids: [B, K, D]
        counts: [B, K]  <-- 新增返回
    """
    B, N, D = x.shape
    device = x.device
    
    if init_centroids is None:
        # 随机初始化
        centroids = torch.stack([
            x[b, torch.randperm(N, device=device)[:num_clusters]] 
            for b in range(B)
        ])
    else:
        centroids = init_centroids

    # 初始化 counts 以防 num_iters=0 (虽然通常不会)
    counts = torch.zeros(B, num_clusters, device=device)
    labels = torch.zeros(B, N, dtype=torch.long, device=device)

    for _ in range(num_iters):
        # 计算距离: ||x-c||^2
        dists = torch.cdist(x, centroids, p=2) 
        
        # E步: 分配标签
        labels = torch.argmin(dists, dim=-1) # [B, N]

        # M步: 更新质心
        mask = F.one_hot(labels, num_classes=num_clusters).float() 
        new_centroids = torch.bmm(mask.transpose(1, 2), x)
        
        # Count: [B, N, K] -> sum -> [B, 1, K]
        counts_keepdim = mask.sum(dim=1, keepdim=True)
        counts = counts_keepdim.squeeze(1) # [B, K]
        
        # 避免除以0
        centroids = new_centroids / (counts_keepdim.transpose(1, 2) + 1e-6)
        
    return labels, centroids, counts

def compute_connectivity_map(q_centroids, k_centroids, k_counts, top_p=0.9):
    """
    计算 SVG2 定义的 Semantically Connected Map (带 Weighted Softmax)。
    Args:
        k_counts: [B, K_k] Key Cluster 的大小
    """
    # 1. 簇间相似度 (Dot Product)
    scale = q_centroids.shape[-1] ** -0.5
    scores = torch.bmm(q_centroids, k_centroids.transpose(1, 2)) * scale # [B, K_q, K_k]
    
    # 2. Weighted Softmax (复现官方逻辑)
    # Prob ~ exp(score) * count
    # log_prob ~ score + log(count)
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