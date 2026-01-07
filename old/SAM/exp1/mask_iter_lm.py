import torch
from typing import Callable, List, Tuple


def predict_sparse_attn_mask_lm(
    B_hw: int,
    num_block_sqrt: int,
    mask_past: torch.Tensor,
    timesteps_past: List[int],
    timestep_now: int,
    alpha: Tuple[float, float, float],
    threshold_inter: float,
    mask_diag_now: torch.Tensor,
    d_disappear: int,
    *,
    num_past_step: int = 2,
    dist_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    优化版：向量化计算稀疏注意力掩码。
    """
    # 参数校验
    assert 1 <= d_disappear <= num_block_sqrt - 1, "d_disappear 超出允许范围"
    assert len(timesteps_past) == num_past_step, "timesteps_past 长度需等于 num_past_step"
    
    device = mask_past.device
    eps = 1e-6
    alpha_w, alpha_h, alpha_t = alpha

    # 1. 预处理输入数据
    # mask_past: [T, N, B, B] -> float32
    mask_past_f = mask_past.to(device=device, dtype=torch.float32)
    
    # 2. 预计算距离
    # 时间距离 d_t: [T]
    t_past_tensor = torch.as_tensor(timesteps_past, device=device, dtype=torch.float32)
    d_t_steps = dist_func(torch.abs(timestep_now - t_past_tensor)).clamp_min(eps)

    # 空间距离表 dist_table: [d_disappear + 1]
    # 涵盖 0 到 d_disappear 的所有整数距离
    dist_indices = torch.arange(d_disappear + 1, device=device, dtype=torch.float32)
    dist_table = dist_func(dist_indices).clamp_min(eps)
    
    dist_0 = dist_table[0] # d=0 的距离值

    # 3. 初始化 4D Grid [N, N, B, B]
    # 使用 uint8 节省显存，最后再 reshape
    grid = torch.zeros(
        (num_block_sqrt, num_block_sqrt, B_hw, B_hw), 
        device=device, 
        dtype=torch.uint8
    )

    # 填充对角线 (d=0)
    # mask_diag_now: [N, B, B]
    diag_indices = torch.arange(num_block_sqrt, device=device)
    grid[diag_indices, diag_indices] = mask_diag_now.to(device=device, dtype=torch.uint8)

    # 4. 向量化计算非对角块
    # 我们按“距离 d”进行迭代，而不是按块坐标迭代
    # 对于固定的 d，所有 (qi, kj) 满足 abs(qi-kj)=d 的块共享相同的权重参数
    
    for d in range(1, d_disappear + 1):
        dist_val = dist_table[d]

        # --- 计算权重向量 [T] ---
        # Row case: d_h=0, d_w=d
        denom_row = alpha_h * dist_0 + alpha_w * dist_val + alpha_t * d_t_steps + eps
        w_row = (1.0 / denom_row).clamp_min(eps) # [T]

        # Col case: d_h=d, d_w=0
        denom_col = alpha_h * dist_val + alpha_w * dist_0 + alpha_t * d_t_steps + eps
        w_col = (1.0 / denom_col).clamp_min(eps) # [T]

        # 归一化因子 (标量)
        # 修改：使用 (w_row + w_col).sum() 以更接近原版 [r0, c0, r1, c1...] 的累加顺序
        norm_factor = (w_row + w_col).sum()

        # --- 加权求和 ---
        # mask_past_f: [T, N, B, B]
        # w_row: [T] -> [T, 1, 1, 1]
        # 结果: [N, B, B]
        weighted_sum_row = (mask_past_f * w_row.view(-1, 1, 1, 1)).sum(dim=0)
        weighted_sum_col = (mask_past_f * w_col.view(-1, 1, 1, 1)).sum(dim=0)

        # --- 填充 Upper Diagonal (kj = qi + d) ---
        # qi 范围: 0 到 N-1-d
        # kj 范围: d 到 N-1
        # Block = (Row_part[qi] + Col_part[kj]) / Norm
        
        # 切片操作，无需循环
        upper_blocks = (weighted_sum_row[:-d] + weighted_sum_col[d:]) / norm_factor
        upper_mask = (upper_blocks > threshold_inter).to(torch.uint8)
        
        # 批量赋值给 Grid
        # grid[qi, qi+d]
        row_indices = torch.arange(num_block_sqrt - d, device=device)
        col_indices = row_indices + d
        grid[row_indices, col_indices] = upper_mask

        # --- 填充 Lower Diagonal (kj = qi - d) ---
        # qi 范围: d 到 N-1
        # kj 范围: 0 到 N-1-d
        # Block = (Row_part[qi] + Col_part[kj]) / Norm
        
        lower_blocks = (weighted_sum_row[d:] + weighted_sum_col[:-d]) / norm_factor
        lower_mask = (lower_blocks > threshold_inter).to(torch.uint8)
        
        row_indices = torch.arange(d, num_block_sqrt, device=device)
        col_indices = row_indices - d
        grid[row_indices, col_indices] = lower_mask

    # 5. 构造 Full Mask
    # Grid [N, N, B, B] -> Permute [N, B, N, B] -> View [N*B, N*B]
    L = num_block_sqrt * B_hw
    full_mask = grid.permute(0, 2, 1, 3).contiguous().view(L, L)

    # 6. 更新历史 Mask
    # mask_now: [T, N, B, B]
    mask_now = torch.empty_like(mask_past)
    if num_past_step > 1:
        mask_now[:-1].copy_(mask_past[1:])
    mask_now[-1].copy_(mask_diag_now)

    return full_mask, mask_now


if __name__ == "__main__":
    # 简单测试以验证运行
    num_past = 2
    B_hw = 2
    num_blocks = 3
    timesteps_hist = [999, 997]
    timestep_cur = 980
    alpha = (1.0, 1.0, 0.2)
    thr_inter = 0.6
    
    mask_hist = torch.tensor(
        [
            [
                [[1, 0], [1, 0]],
                [[0, 1], [0, 0]],
                [[1, 1], [1, 0]],
            ],
            [
                [[0, 1], [1, 1]],
                [[0, 0], [1, 1]],
                [[1, 1], [1, 0]],
            ],
        ],
        dtype=torch.uint8,
    )

    mask_diag_cur = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[1, 1], [0, 1]],
        ],
        dtype=torch.uint8,
    )

    full_mask, mask_next = predict_sparse_attn_mask_lm(
        B_hw=B_hw,
        num_block_sqrt=num_blocks,
        mask_past=mask_hist,
        timesteps_past=timesteps_hist,
        timestep_now=timestep_cur,
        alpha=alpha,
        threshold_inter=thr_inter,
        mask_diag_now=mask_diag_cur,
        d_disappear=1,
        num_past_step=num_past,
    )

    print("Full mask shape:", full_mask.shape)
    print(full_mask)
    print("\nNext history shape:", mask_next.shape)
    print(mask_next)