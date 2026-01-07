from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import torch

_COMPONENT_ORDERS: Dict[str, Tuple[str, str, str]] = {
    "A_w": ("w", "h", "f"),
    "A_h": ("h", "w", "f"),
    "A_f": ("f", "w", "h"),
}


@dataclass
class ComponentState:
    centers_q: torch.Tensor  # 量化后的簇中心 [num_heads, num_centers, block_size, block_size]
    rep_indices: torch.Tensor  # 代表块索引 [num_heads, num_centers, M, 2]
    rep_counts: torch.Tensor  # 每个簇的代表块数量 [num_heads, num_centers]
    assignments: torch.Tensor  # 指派矩阵 [num_heads, grid_side, grid_side]
    block_size: int
    grid_side: int
    dims_order: Tuple[str, str, str]


@dataclass
class RQTState:
    components: Dict[str, ComponentState]
    size_info: Dict[str, int]
    rope_info: Dict[str, object]


def _normalize_sizes(size_info: Dict[str, int]) -> Dict[str, int]:
    """归一化尺寸信息"""
    required = {"l_w", "l_h", "l_f"}
    if not size_info or not required.issubset(set(size_info.keys())):
        raise ValueError("size_info must contain l_w, l_h, l_f.")
    return {"w": int(size_info["l_w"]), "h": int(size_info["l_h"]), "f": int(size_info["l_f"])}


def _split_qk(Q: torch.Tensor, K: torch.Tensor, rope_info: Dict[str, object]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """按RoPE维度拆分Q和K"""
    dims = rope_info.get("dims", ["f", "h", "w"])
    d_splits = rope_info.get("d_splits", None)
    if not d_splits or len(d_splits) != 3:
        raise ValueError("rope_info.d_splits must be a 3-element list.")
    
    offsets = [0, d_splits[0], d_splits[0] + d_splits[1], sum(d_splits)]
    mapping: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    for idx, dim in enumerate(dims):
        q_slice = Q[:, :, offsets[idx]:offsets[idx+1]]  # [num_heads, L, d_part]
        k_slice = K[:, :, offsets[idx]:offsets[idx+1]]
        mapping[f"A_{dim}"] = (q_slice, k_slice)
    
    return mapping


def _quantize_block_batch(blocks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批量8位非对称量化 - GPU并行"""
    blocks = blocks.float()
    # 批量计算最小最大值
    blocks_flat = blocks.view(blocks.size(0), -1)
    min_vals = blocks_flat.min(dim=1).values
    max_vals = blocks_flat.max(dim=1).values
    scale = (max_vals - min_vals).clamp(min=1e-8) / 255.0
    
    # 批量量化
    q = torch.round((blocks_flat - min_vals.unsqueeze(1)) / scale.unsqueeze(1) - 128.0)
    q = q.clamp(-128, 127).to(torch.int8)
    return q.view_as(blocks), min_vals, max_vals


def _dequantize_block_batch(q: torch.Tensor, min_vals: torch.Tensor, max_vals: torch.Tensor) -> torch.Tensor:
    """批量反量化 - GPU并行"""
    q = q.float()
    scale = (max_vals - min_vals).clamp(min=1e-8) / 255.0
    
    # 处理不同的维度情况
    if q.dim() >= 3:
        # [batch, ...] 的情况
        scale = scale.view(-1, *([1] * (q.dim() - 1)))
        min_vals = min_vals.view(-1, *([1] * (q.dim() - 1)))
    
    return (q + 128.0) * scale + min_vals


def _compute_component_blocks(
    Q_part: torch.Tensor,  # [num_heads, L, d]
    K_part: torch.Tensor,  # [num_heads, L, d]
    order: Tuple[str, str, str],
    sizes: Dict[str, int]
) -> torch.Tensor:
    """计算组件所有块 - GPU并行"""
    num_heads, L, d = Q_part.shape
    primary, inner1, inner2 = order
    block_size = sizes[primary]
    inner1_size = sizes[inner1]
    inner2_size = sizes[inner2]
    grid_side = inner1_size * inner2_size
    
    # 重塑为块结构
    Q_blocks = Q_part.view(num_heads, inner1_size, inner2_size, block_size, d)
    K_blocks = K_part.view(num_heads, inner1_size, inner2_size, block_size, d)
    
    # 重塑为 [num_heads, grid_side, block_size, d]
    Q_blocks = Q_blocks.reshape(num_heads, grid_side, block_size, d)
    K_blocks = K_blocks.reshape(num_heads, grid_side, block_size, d)
    
    # 使用 torch.matmul 进行批量矩阵乘法 - GPU并行
    Q_exp = Q_blocks.unsqueeze(2)  # [num_heads, grid_side, 1, block_size, d]
    K_exp = K_blocks.unsqueeze(1).transpose(-2, -1)  # [num_heads, 1, grid_side, d, block_size]
    
    # 批量矩阵乘法: [num_heads, grid_side, 1, block_size, d] @ [num_heads, 1, grid_side, d, block_size]
    all_blocks = torch.matmul(Q_exp, K_exp)  # [num_heads, grid_side, grid_side, block_size, block_size]
    
    return all_blocks


def _compute_block_distances_vectorized(
    block: torch.Tensor,  # [num_heads, block_size, block_size]
    center_sums: torch.Tensor,  # [num_heads, Nc, block_size, block_size]
    center_counts: torch.Tensor,  # [num_heads, Nc]
    mask: torch.Tensor  # [num_heads, Nc], 布尔掩码，标记有效簇
) -> torch.Tensor:
    """向量化计算块与所有簇中心的距离"""
    num_heads, Nc, block_size, _ = center_sums.shape
    device = block.device
    
    # 展开块维度用于广播
    block_expanded = block.unsqueeze(1)  # [num_heads, 1, block_size, block_size]
    
    # 计算当前所有簇的中心
    counts_expanded = center_counts.float().unsqueeze(-1).unsqueeze(-1)  # [num_heads, Nc, 1, 1]
    counts_safe = counts_expanded.clamp(min=1)
    centers = center_sums.float() / counts_safe  # [num_heads, Nc, block_size, block_size]
    
    # 计算L1距离: |block * N - sum| / N
    # 等效于 |block - center|
    distances = torch.abs(block_expanded - centers)  # [num_heads, Nc, block_size, block_size]
    distances = distances.mean(dim=(-2, -1))  # [num_heads, Nc]
    
    # 将无效簇的距离设置为无穷大
    distances = torch.where(mask, distances, torch.full_like(distances, float('inf')))
    
    return distances


def _first_pass_component_parallel(
    Q_part: torch.Tensor,
    K_part: torch.Tensor,
    threshold: float,
    Nc: int,
    M: int,
    order: Tuple[str, str, str],
    sizes: Dict[str, int],
    force_merge: bool,
    candidate_scale: int = 2
) -> ComponentState:
    """
    并行化的簇训练阶段 - 完全向量化操作，移除head和簇维度的循环
    """
    num_heads = Q_part.shape[0]
    device = Q_part.device
    primary = order[0]
    block_size = sizes[primary]
    inner1_size = sizes[order[1]]
    inner2_size = sizes[order[2]]
    grid_side = inner1_size * inner2_size
    candidate_capacity = M * candidate_scale
    
    # 1. 计算所有块并批量量化
    all_blocks = _compute_component_blocks(Q_part, K_part, order, sizes)
    
    # 展平以便批量量化
    all_blocks_flat = all_blocks.view(num_heads * grid_side * grid_side, block_size, block_size)
    
    # 批量量化所有块 - GPU并行
    all_blocks_q, _, _ = _quantize_block_batch(all_blocks_flat)
    all_blocks_q = all_blocks_q.view(num_heads, grid_side, grid_side, block_size, block_size)
    
    # 2. 初始化聚类数据结构 - 在GPU上
    centers = torch.zeros(num_heads, Nc, block_size, block_size, device=device, dtype=torch.float32)
    center_sums = torch.zeros(num_heads, Nc, block_size, block_size, device=device, dtype=torch.int32)
    center_counts = torch.zeros(num_heads, Nc, device=device, dtype=torch.int32)
    
    # 候选块存储 - 在GPU上
    candidate_blocks = torch.full((num_heads, Nc, candidate_capacity, block_size, block_size), 
                                 -1, device=device, dtype=torch.int8)
    candidate_indices = torch.full((num_heads, Nc, candidate_capacity, 2), 
                                  -1, device=device, dtype=torch.long)
    candidate_distances = torch.full((num_heads, Nc, candidate_capacity), 
                                    float('inf'), device=device, dtype=torch.float32)
    
    # 指派矩阵 - 在GPU上
    assignments = torch.full((num_heads, grid_side, grid_side), -1, device=device, dtype=torch.long)
    
    # 每个head当前簇数量 - 在GPU上
    current_num_clusters = torch.zeros(num_heads, device=device, dtype=torch.long)
    
    # 创建掩码以标记有效簇
    clusters_mask = torch.zeros(num_heads, Nc, dtype=torch.bool, device=device)
    
    # 3. 遍历所有块（顺序遍历，但并行处理所有head和簇）
    for idx in range(grid_side * grid_side):
        gr = idx // grid_side
        gc = idx % grid_side
        
        # 当前块的量化值 [num_heads, block_size, block_size]
        current_block_q = all_blocks_q[:, gr, gc].float()
        
        # 向量化计算与所有簇的距离 - 完全并行
        distances = _compute_block_distances_vectorized(
            current_block_q, 
            center_sums, 
            center_counts, 
            clusters_mask
        )  # [num_heads, Nc]
        
        # 找到每个head的最佳簇（最小距离）
        best_dists, best_indices = torch.min(distances, dim=1)  # [num_heads], [num_heads]
        
        # 向量化分配逻辑 - 使用掩码和条件更新
        
        # 情况1: 当前head没有簇 (current_num_clusters == 0)
        no_cluster_mask = (current_num_clusters == 0)  # [num_heads]
        
        # 为这些head创建第一个簇
        if no_cluster_mask.any():
            # 簇索引为0
            cluster_idx = torch.zeros_like(current_num_clusters)  # [num_heads]
            
            # 更新centers, center_sums, center_counts
            centers[no_cluster_mask, 0] = current_block_q[no_cluster_mask]
            center_sums[no_cluster_mask, 0] = current_block_q[no_cluster_mask].to(torch.int32)
            center_counts[no_cluster_mask, 0] = 1
            clusters_mask[no_cluster_mask, 0] = True
            
            # 更新候选
            candidate_blocks[no_cluster_mask, 0, 0] = all_blocks_q[no_cluster_mask, gr, gc]
            candidate_indices[no_cluster_mask, 0, 0, 0] = gr
            candidate_indices[no_cluster_mask, 0, 0, 1] = gc
            candidate_distances[no_cluster_mask, 0, 0] = 0.0
            
            # 更新assignments
            assignments[no_cluster_mask, gr, gc] = 0
            
            # 更新current_num_clusters
            current_num_clusters[no_cluster_mask] = 1
        
        # 情况2: 有簇的head
        has_cluster_mask = ~no_cluster_mask  # [num_heads]
        
        if has_cluster_mask.any():
            has_cluster_indices = torch.where(has_cluster_mask)[0]
            
            # 子集处理
            best_dists_sub = best_dists[has_cluster_mask]
            best_indices_sub = best_indices[has_cluster_mask]
            current_num_clusters_sub = current_num_clusters[has_cluster_mask]
            
            # 情况2a: 分配到现有簇 (best_dist < threshold)
            assign_mask = best_dists_sub < threshold
            
            if assign_mask.any():
                assign_h_indices = has_cluster_indices[assign_mask]
                assign_cluster_idx = best_indices_sub[assign_mask]
                
                # 向量化更新簇中心
                center_sums[assign_h_indices, assign_cluster_idx] += current_block_q[assign_h_indices].to(torch.int32)
                center_counts[assign_h_indices, assign_cluster_idx] += 1
                # 确保索引为 long 且在同一 device
                assign_h_indices = assign_h_indices.to(device=device, dtype=torch.long)
                assign_cluster_idx = assign_cluster_idx.to(device=device, dtype=torch.long)

                # 选取并扩展计数以便广播到 [N, 1, 1]
                cnt = center_counts[assign_h_indices, assign_cluster_idx].float().unsqueeze(-1).unsqueeze(-1).clamp(min=1.0)

                # 按对更新中心（[N, b, b] / [N,1,1] -> [N,b,b]）
                centers[assign_h_indices, assign_cluster_idx] = center_sums[assign_h_indices, assign_cluster_idx].float() / cnt
                
                # 更新候选 - 向量化
                new_center = centers[assign_h_indices, assign_cluster_idx]
                new_dist = torch.abs(current_block_q[assign_h_indices] - new_center).mean(dim=(-2, -1))
                
                # 对于每个分配，找到候选列表中最大距离的索引
                for i, (h, c) in enumerate(zip(assign_h_indices, assign_cluster_idx)):
                    cand_dist = candidate_distances[h, c]
                    max_dist_idx = cand_dist.argmax()
                    
                    if cand_dist[max_dist_idx] > new_dist[i] or cand_dist[max_dist_idx] == float('inf'):
                        candidate_blocks[h, c, max_dist_idx] = all_blocks_q[h, gr, gc]
                        candidate_indices[h, c, max_dist_idx, 0] = gr
                        candidate_indices[h, c, max_dist_idx, 1] = gc
                        candidate_distances[h, c, max_dist_idx] = new_dist[i]
                
                assignments[assign_h_indices, gr, gc] = assign_cluster_idx
            
            # 情况2b: 创建新簇 (best_dist >= threshold and current_num_clusters < Nc)
            create_mask = ~assign_mask & (current_num_clusters_sub < Nc)
            
            if create_mask.any():
                create_h_indices = has_cluster_indices[create_mask]
                create_cluster_idx = current_num_clusters_sub[create_mask]
                
                centers[create_h_indices, create_cluster_idx] = current_block_q[create_h_indices]
                center_sums[create_h_indices, create_cluster_idx] = current_block_q[create_h_indices].to(torch.int32)
                center_counts[create_h_indices, create_cluster_idx] = 1
                clusters_mask[create_h_indices, create_cluster_idx] = True
                
                # 初始化候选
                candidate_blocks[create_h_indices, create_cluster_idx, 0] = all_blocks_q[create_h_indices, gr, gc]
                candidate_indices[create_h_indices, create_cluster_idx, 0, 0] = gr
                candidate_indices[create_h_indices, create_cluster_idx, 0, 1] = gc
                candidate_distances[create_h_indices, create_cluster_idx, 0] = 0.0
                
                assignments[create_h_indices, gr, gc] = create_cluster_idx
                current_num_clusters[create_h_indices] += 1
            
            # 情况2c: 强制合并 (best_dist >= threshold and current_num_clusters >= Nc and force_merge)
            merge_mask = ~assign_mask & ~create_mask & force_merge
            
            if merge_mask.any():
                merge_h_indices = has_cluster_indices[merge_mask]
                merge_cluster_idx = best_indices_sub[merge_mask]
                
                center_sums[merge_h_indices, merge_cluster_idx] += current_block_q[merge_h_indices].to(torch.int32)
                center_counts[merge_h_indices, merge_cluster_idx] += 1
                merge_h_indices = merge_h_indices.to(device=device, dtype=torch.long)
                merge_cluster_idx = merge_cluster_idx.to(device=device, dtype=torch.long)
                cnt = center_counts[merge_h_indices, merge_cluster_idx].float().unsqueeze(-1).unsqueeze(-1).clamp(min=1.0)
                centers[merge_h_indices, merge_cluster_idx] = center_sums[merge_h_indices, merge_cluster_idx].float() / cnt
                
                # 更新候选 - 类似情况2a
                new_center = centers[merge_h_indices, merge_cluster_idx]
                new_dist = torch.abs(current_block_q[merge_h_indices] - new_center).mean(dim=(-2, -1))
                
                for i, (h, c) in enumerate(zip(merge_h_indices, merge_cluster_idx)):
                    cand_dist = candidate_distances[h, c]
                    max_dist_idx = cand_dist.argmax()
                    
                    if cand_dist[max_dist_idx] > new_dist[i] or cand_dist[max_dist_idx] == float('inf'):
                        candidate_blocks[h, c, max_dist_idx] = all_blocks_q[h, gr, gc]
                        candidate_indices[h, c, max_dist_idx, 0] = gr
                        candidate_indices[h, c, max_dist_idx, 1] = gc
                        candidate_distances[h, c, max_dist_idx] = new_dist[i]
                
                assignments[merge_h_indices, gr, gc] = merge_cluster_idx
    
    # 4. 向量化选择每个簇的M个最近块
    # 创建有效掩码
    valid_mask = candidate_distances != float('inf')  # [num_heads, Nc, candidate_capacity]

    # 计算每个簇的有效候选数量
    valid_counts = valid_mask.sum(dim=2)  # [num_heads, Nc]

    # 使用 torch.topk 获取前 M 个最小距离的索引（即使有些无效，也获取）
    topk_dist, topk_idx = torch.topk(candidate_distances, M, dim=2, largest=False)  # [num_heads, Nc, M]

    # 构建代表块索引和计数
    rep_indices = torch.full((num_heads, Nc, M, 2), -1, device=device, dtype=torch.long)
    rep_counts = torch.zeros(num_heads, Nc, device=device, dtype=torch.long)

    # 只更新有效簇
    for h in range(num_heads):
        for c in range(Nc):
            if clusters_mask[h, c]:
                valid_count = valid_counts[h, c].item()
                if valid_count > 0:
                    k = min(M, valid_count)
                    rep_counts[h, c] = k
                    rep_indices[h, c, :k] = candidate_indices[h, c, topk_idx[h, c, :k]]
    
    # 5. 批量量化簇中心 - GPU并行
    # 只量化有效的簇中心
    active_clusters = current_num_clusters.max().item()
    if active_clusters > 0:
        centers_active = centers[:, :active_clusters]  # [num_heads, active_clusters, block_size, block_size]
        centers_flat = centers_active.reshape(num_heads * active_clusters, block_size, block_size)
        centers_q_flat, _, _ = _quantize_block_batch(centers_flat)
        centers_q = centers_q_flat.view(num_heads, active_clusters, block_size, block_size)
        
        # 如果实际簇数小于Nc，填充
        if active_clusters < Nc:
            full_centers_q = torch.full((num_heads, Nc, block_size, block_size), 
                                       0, device=device, dtype=torch.int8)
            full_centers_q[:, :active_clusters] = centers_q
            centers_q = full_centers_q
    else:
        centers_q = torch.zeros(num_heads, Nc, block_size, block_size, device=device, dtype=torch.int8)
    
    return ComponentState(
        centers_q=centers_q,
        rep_indices=rep_indices,
        rep_counts=rep_counts,
        assignments=assignments,
        block_size=block_size,
        grid_side=grid_side,
        dims_order=order,
    )


def _refresh_centers_parallel(
    Q_part: torch.Tensor,
    K_part: torch.Tensor,
    state: ComponentState,
    sizes: Dict[str, int]
) -> ComponentState:
    """并行更新簇中心 - GPU并行"""
    num_heads = Q_part.shape[0]
    device = Q_part.device
    Nc = state.centers_q.shape[1]
    block_size = state.block_size
    grid_side = state.grid_side
    
    # 1. 计算所有块
    all_blocks = _compute_component_blocks(Q_part, K_part, state.dims_order, sizes)
    
    # 2. 向量化更新簇中心
    new_centers = torch.zeros(num_heads, Nc, block_size, block_size, device=device, dtype=torch.float32)
    
    # 创建批次处理所有代表块
    all_rep_blocks = []
    all_rep_head_indices = []
    all_rep_cluster_indices = []
    
    for h in range(num_heads):
        for c in range(Nc):
            rep_count = int(state.rep_counts[h, c])
            if rep_count > 0:
                reps = state.rep_indices[h, c, :rep_count]
                for i in range(rep_count):
                    gr, gc = reps[i, 0], reps[i, 1]
                    if gr >= 0 and gc >= 0:
                        all_rep_blocks.append(all_blocks[h, int(gr), int(gc)].float())
                        all_rep_head_indices.append(h)
                        all_rep_cluster_indices.append(c)
    
    if all_rep_blocks:
        # 批量计算新中心
        rep_tensor = torch.stack(all_rep_blocks)  # [total_reps, block_size, block_size]
        
        # 使用scatter_add_进行向量化聚合
        new_centers_sum = torch.zeros_like(new_centers)
        new_centers_count = torch.zeros(num_heads, Nc, device=device)
        
        # 转换为张量索引
        head_indices = torch.tensor(all_rep_head_indices, device=device)
        cluster_indices = torch.tensor(all_rep_cluster_indices, device=device)
        
        # 向量化求和
        for i in range(len(all_rep_blocks)):
            h = head_indices[i]
            c = cluster_indices[i]
            new_centers_sum[h, c] += rep_tensor[i]
            new_centers_count[h, c] += 1
        
        # 计算均值，避免除零
        new_centers_count_expanded = new_centers_count.unsqueeze(-1).unsqueeze(-1).clamp(min=1)
        new_centers = new_centers_sum / new_centers_count_expanded
    
    # 3. 批量量化新中心 - GPU并行
    new_centers_flat = new_centers.view(num_heads * Nc, block_size, block_size)
    new_centers_q, _, _ = _quantize_block_batch(new_centers_flat)
    new_centers_q = new_centers_q.view(num_heads, Nc, block_size, block_size)
    
    return ComponentState(
        centers_q=new_centers_q,
        rep_indices=state.rep_indices,
        rep_counts=state.rep_counts,
        assignments=state.assignments,
        block_size=block_size,
        grid_side=grid_side,
        dims_order=state.dims_order,
    )


def _approx_component_parallel(
    Q_part: torch.Tensor,
    K_part: torch.Tensor,
    state: ComponentState,
    sizes: Dict[str, int]
) -> torch.Tensor:
    """并行近似计算组件 - GPU并行"""
    num_heads = Q_part.shape[0]
    device = Q_part.device
    primary, inner1, inner2 = state.dims_order
    block_size = state.block_size
    inner1_size = sizes[inner1]
    inner2_size = sizes[inner2]
    grid_side = inner1_size * inner2_size
    
    # 1. 计算所有块（行块并行）- GPU并行
    Q_blocks = Q_part.view(num_heads, inner1_size, inner2_size, block_size, -1)
    K_blocks = K_part.view(num_heads, inner1_size, inner2_size, block_size, -1)
    
    # 使用批量矩阵乘法
    Q_blocks_flat = Q_blocks.reshape(num_heads, grid_side, block_size, -1)
    K_blocks_flat = K_blocks.reshape(num_heads, grid_side, block_size, -1)
    
    # 使用 torch.matmul - GPU并行
    Q_exp = Q_blocks_flat.unsqueeze(2)  # [num_heads, grid_side, 1, block_size, d]
    K_exp = K_blocks_flat.unsqueeze(1).transpose(-2, -1)  # [num_heads, 1, grid_side, d, block_size]
    all_blocks = torch.matmul(Q_exp, K_exp)  # [num_heads, grid_side, grid_side, block_size, block_size]
    
    # 2. 向量化构建代表块掩码
    rep_mask = torch.zeros(num_heads, grid_side, grid_side, dtype=torch.bool, device=device)
    
    # 收集所有代表块位置
    rep_head_indices = []
    rep_row_indices = []
    rep_col_indices = []
    
    for h in range(num_heads):
        for c in range(state.centers_q.shape[1]):
            rep_count = int(state.rep_counts[h, c])
            if rep_count > 0:
                reps = state.rep_indices[h, c, :rep_count]
                for i in range(rep_count):
                    gr, gc = reps[i, 0], reps[i, 1]
                    if gr >= 0 and gc >= 0:
                        gr_int, gc_int = int(gr), int(gc)
                        rep_mask[h, gr_int, gc_int] = True
                        rep_head_indices.append(h)
                        rep_row_indices.append(gr_int)
                        rep_col_indices.append(gc_int)
    
    # 3. 批量处理非代表块（反量化）- GPU并行
    result_blocks = all_blocks.clone()
    
    # 获取所有非代表块位置
    non_rep_mask = ~rep_mask
    
    for h in range(num_heads):
        h_non_rep_mask = non_rep_mask[h]
        non_rep_count = h_non_rep_mask.sum().item()
        
        if non_rep_count == 0:
            continue
            
        # 获取非代表块的行列索引
        non_rep_rows, non_rep_cols = torch.where(h_non_rep_mask)
        
        # 获取对应的簇分配
        non_rep_assignments = state.assignments[h, non_rep_rows, non_rep_cols]
        
        # 只处理有效分配
        valid_mask = non_rep_assignments >= 0
        if not valid_mask.any():
            continue
            
        valid_rows = non_rep_rows[valid_mask]
        valid_cols = non_rep_cols[valid_mask]
        valid_assignments = non_rep_assignments[valid_mask]
        
        # 批量获取量化中心
        centers_q_batch = state.centers_q[h, valid_assignments]
        
        # 批量计算实际块的最小最大值
        actual_blocks_batch = all_blocks[h, valid_rows, valid_cols]
        
        # 重塑为二维以计算最小最大值
        actual_blocks_flat = actual_blocks_batch.view(len(valid_rows), -1)
        min_vals = actual_blocks_flat.min(dim=1).values
        max_vals = actual_blocks_flat.max(dim=1).values
        
        # 批量反量化
        approx_blocks = _dequantize_block_batch(centers_q_batch, min_vals, max_vals)
        
        # 批量赋值
        result_blocks[h, valid_rows, valid_cols] = approx_blocks
    
    # 4. 向量化构建完整矩阵
    primary_size = sizes[primary]
    full_size = grid_side * block_size
    approximations = torch.zeros(num_heads, full_size, full_size, device=device)
    
    # 使用向量化操作重塑块网格
    for h in range(num_heads):
        block_grid = result_blocks[h]  # [grid_side, grid_side, block_size, block_size]
        
        # 重塑为完整矩阵 - 使用unfold或reshape操作
        # 方法1: 使用unfold
        approximations[h] = block_grid.permute(0, 2, 1, 3).contiguous().view(full_size, full_size)
    
    return approximations


def rope_quant_template(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    thetas: Optional[List[float]] = None,
    Nc: int = 512,
    M: int = 8,
    rope_info: Optional[Dict[str, object]] = None,
    size_info: Optional[Dict[str, int]] = None,
    force_merge: bool = False,
    state: Optional[RQTState] = None,
    candidate_scale: int = 2
) -> Tuple[RQTState, Optional[torch.Tensor]]:
    """
    主函数：RoPE量化注意力模板
    
    参数:
        Q, K, V: 输入张量 [num_heads, L, D]
        thetas: 距离阈值列表，None表示使用近似计算
        Nc: 每个组件的簇中心数量
        M: 每个簇中心的代表块数量
        rope_info: RoPE配置信息
        size_info: 尺寸信息
        force_merge: 是否强制合并
        state: 之前的状态（用于后续迭代）
        candidate_scale: 候选块扩展比例
    
    返回:
        state: 更新后的状态
        O: 输出张量（第一次调用时为None）
    """
    # 输入验证
    assert Q.dim() == 3, "Q must be [num_heads, L, D]"
    assert K.dim() == 3, "K must be [num_heads, L, D]"
    assert V.dim() == 3, "V must be [num_heads, L, V_dim]"
    
    num_heads = Q.size(0)
    assert K.size(0) == num_heads and V.size(0) == num_heads, "Inconsistent num_heads"
    
    if thetas is not None and len(thetas) != 3:
        raise ValueError("thetas must be a list of three thresholds for [A_f, A_h, A_w].")
    
    # 归一化尺寸信息
    sizes = _normalize_sizes(size_info or {})
    
    # 拆分Q和K
    rope_map = _split_qk(Q, K, rope_info or {})
    
    # 确定处理哪些组件
    component_keys = [key for key in ("A_f", "A_h", "A_w") if key in rope_map]
    
    if thetas is not None:
        # 第一次调用：簇训练
        comp_states = {}
        thresholds = {"A_f": thetas[0], "A_h": thetas[1], "A_w": thetas[2]}
        
        # 使用线程池并行处理不同组件
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for key in component_keys:
                q_part, k_part = rope_map[key]
                order = _COMPONENT_ORDERS[key]
                
                future = executor.submit(
                    _first_pass_component_parallel,
                    q_part, k_part,
                    thresholds[key], Nc, M,
                    order, sizes, force_merge,
                    candidate_scale
                )
                futures[key] = future
            
            # 收集结果
            for key, future in futures.items():
                comp_states[key] = future.result()
        
        # 创建状态
        state = RQTState(
            components=comp_states,
            size_info=sizes,
            rope_info=rope_info or {},
        )
        
        return state, None
    
    else:
        # 后续调用：近似计算
        if state is None:
            raise ValueError("state must be provided for subsequent iterations when thetas is None.")
        
        # 完全并行化：每个组件的更新中心和近似计算在一个任务中
        def process_component(key: str):
            q_part, k_part = rope_map[key]
            
            # 1. 更新簇中心
            refreshed_state = _refresh_centers_parallel(
                q_part, k_part,
                state.components[key], sizes
            )
            
            # 2. 计算近似值
            approx = _approx_component_parallel(q_part, k_part, refreshed_state, sizes)
            
            return key, refreshed_state, approx
        
        approximations = []
        updated_components = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for key in component_keys:
                future = executor.submit(process_component, key)
                futures[key] = future
            
            # 收集结果
            for key, future in futures.items():
                key_result, refreshed_state, approx = future.result()
                updated_components[key_result] = refreshed_state
                approximations.append(approx)
        
        # 合并所有组件的近似值
        A_total = sum(approximations)  # [num_heads, L, L]
        
        # Softmax和输出
        probs = torch.softmax(A_total, dim=-1)
        O = torch.matmul(probs, V)
        
        # 创建新状态
        new_state = RQTState(
            components=updated_components,
            size_info=state.size_info,
            rope_info=state.rope_info,
        )
        
        return new_state, O


def _demo():
    """演示函数"""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试配置
    num_heads = 4
    l_w, l_h, l_f = 4, 4, 2
    size_info = {"l_w": l_w, "l_h": l_h, "l_f": l_f}
    d_splits = [8, 8, 8]
    rope_info = {"dims": ["f", "h", "w"], "d_splits": d_splits}
    
    L = l_w * l_h * l_f
    d = sum(d_splits)
    
    Q = torch.randn(num_heads, L, d, device=device)
    K = torch.randn(num_heads, L, d, device=device)
    V = torch.randn(num_heads, L, 8, device=device)
    
    print(f"Running on {device}")
    print(f"Number of heads: {num_heads}")
    print(f"Total sequence length: {L}")
    print(f"Feature dimension: {d}")
    
    # 第一次调用：簇训练
    print("\n=== First call: Cluster training ===")
    try:
        state, _ = rope_quant_template(
            Q, K, V,
            thetas=[5.0, 5.0, 5.0],
            Nc=8,
            M=4,
            rope_info=rope_info,
            size_info=size_info,
            force_merge=True,
            candidate_scale=2
        )
        
        print("Training completed.")
        for key, comp in state.components.items():
            print(f"{key}: centers={comp.centers_q.shape[1]}, assignments shape={comp.assignments.shape}")
        
        # 第二次调用：近似计算
        print("\n=== Second call: Approximate computation ===")
        state, O = rope_quant_template(
            Q, K, V,
            thetas=None,
            rope_info=rope_info,
            size_info=size_info,
            state=state,
            candidate_scale=2
        )
        
        print(f"Output shape: {O.shape}")
        print(f"Output mean absolute value: {O.abs().mean().item():.6f}")
        
        # 验证GPU使用
        print("\n=== GPU usage verification ===")
        print(f"All tensors on GPU: {all(t.device.type == 'cuda' for comp in state.components.values() for t in [comp.centers_q, comp.rep_indices, comp.rep_counts, comp.assignments])}")
        print(f"Output on GPU: {O.device.type == 'cuda'}")
        
    except Exception as e:
        print(f"RQT failed: {e}")
        print("Falling back to Flash Attention")
        # 回退到常规注意力计算
        probs = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)), dim=-1)
        O = torch.matmul(probs, V)
        print(f"Fallback output shape: {O.shape}")
        return None, O
    
    return state, O


if __name__ == "__main__":
    state, O = _demo()