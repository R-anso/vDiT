import math
import torch


def generate_diff_sparse_mask_lm(
    tensor: torch.Tensor,
    B_h: int,
    B_w: int,
    alpha: float,
    beta: float,
    *,
    mode: str = "ndiff",
) -> torch.Tensor:
    """
    低显存版本：以最小的中间张量开销生成稀疏掩码（无日志）。
    """
    if tensor.ndim != 3:
        raise ValueError("tensor 需为 3 维张量 [Batch, B_hw, B_hw]")
    batch_size, num_rows, num_cols = tensor.shape
    if num_rows != num_cols:
        raise ValueError(f"输入需为方阵形式，得到 ({num_rows}, {num_cols})")
    if B_h <= 0 or B_w <= 0:
        raise ValueError("B_h 与 B_w 需为正整数")
    if B_h * B_w != num_cols:
        raise ValueError(f"B_h * B_w 应等于列数 {num_cols}")
    if alpha <= 0:
        raise ValueError("alpha 必须为正数")
    if not (0.0 <= beta <= 0.5):
        raise ValueError("beta 需位于 [0, 0.5]")

    mode = mode.lower()
    if mode not in {"ndiff", "topk"}:
        raise ValueError(f"mode 需为 'ndiff' 或 'topk'，得到 {mode}")

    # 优化1: 保持 float32 进行核心计算以对齐精度，仅在非关键路径降级
    # 如果原始输入就是 fp16，为了对齐原版(原版强转fp32)，这里也先转fp32
    values = tensor.to(torch.float32) 
    result = torch.zeros((batch_size, num_rows, num_cols), device=values.device, dtype=torch.bool)

    if mode == "ndiff":
        # 使用 view 而不是 reshape 尝试避免内存拷贝（如果内存连续）
        segments = values.view(batch_size, num_rows, B_h, B_w)

        # 这里的计算必须是 float32 才能和原版对齐
        segment_max = segments.max(dim=-1).values
        row_max = segment_max.max(dim=-1, keepdim=True).values
        diff = row_max - segment_max

        # Grades 使用 int8 足够
        grades = torch.zeros_like(diff, dtype=torch.int8)
        grades.masked_fill_(diff <= alpha, 3)
        grades.masked_fill_((diff > alpha) & (diff <= 2 * alpha), 2)
        grades.masked_fill_((diff > 2 * alpha) & (diff <= 3 * alpha), 1)

        # 计算 keep_counts
        # 注意：原版是 float(beta)，这里保持一致
        keep_counts = torch.ceil(grades.float() * (beta * B_w)).to(torch.int32)
        keep_counts.clamp_(max=B_w)

        max_keep = int(keep_counts.max().item())
        if max_keep == 0:
            return result

        # 优化2: 使用 topk 替代 argsort，但在 float32 下进行以保证比较结果一致
        # sorted=True 有助于在某些并列情况下保持稍微确定的行为，虽然不保证完全一致
        topk_indices = torch.topk(segments, k=max_keep, dim=-1, largest=True, sorted=True).indices
        
        rank_range = torch.arange(max_keep, device=segments.device, dtype=keep_counts.dtype)
        valid_mask = rank_range.view(1, 1, 1, max_keep) < keep_counts.unsqueeze(-1)

        segment_view = result.view(batch_size, num_rows, B_h, B_w)
        segment_view.scatter_(-1, topk_indices, valid_mask)
        return result

    keep_elems = int(math.ceil(beta * num_cols))
    if keep_elems <= 0:
        return result

    keep_elems = min(keep_elems, num_cols)
    topk_indices = torch.topk(values, k=keep_elems, dim=-1, largest=True, sorted=True).indices
    result.scatter_(-1, topk_indices, True)
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    B_h, B_w = 2, 3
    alpha = 0.05
    beta = 0.25
    sample = torch.tensor(
        [
            [
                [0.9, 0.6, 0.1, 0.8, 0.3, 0.2],
                [0.5, 0.4, 0.2, 0.7, 0.6, 0.1],
                [0.3, 0.2, 0.1, 0.4, 0.3, 0.2],
                [0.8, 0.7, 0.6, 0.9, 0.8, 0.7],
                [0.1, 0.2, 0.1, 0.3, 0.2, 0.1],
                [0.6, 0.5, 0.4, 0.7, 0.6, 0.5],
            ]
        ],
        dtype=torch.float32,
    )
    mask = generate_diff_sparse_mask_lm(sample, B_h=B_h, B_w=B_w, alpha=alpha, beta=beta, mode="topk")
    print("输入：", sample)
    print("掩码：", mask.to(dtype=torch.uint8))