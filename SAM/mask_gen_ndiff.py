import os
import time
import numpy as np
import torch


def _ensure_log_dir(log_dir: str) -> None:
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def _log_run_summary(
    log_enabled: bool,
    log_dir: str,
    *,
    batch: int,
    B_h: int,
    B_w: int,
    alpha: float,
) -> None:
    if not log_enabled:
        return
    _ensure_log_dir(log_dir)
    summary_path = os.path.join(log_dir, "diff_mask_run_summary.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            f"[{timestamp}] generate_diff_sparse_mask call:\n"
            f"  batch = {batch}\n"
            f"  B_h = {B_h}\n"
            f"  B_w = {B_w}\n"
            f"  alpha = {alpha}\n\n"
        )


def _maybe_log_matrix(
    log_enabled: bool,
    log_dir: str,
    name: str,
    tensor: torch.Tensor,
    extra_info: str = "",
) -> None:
    if not log_enabled:
        return
    _ensure_log_dir(log_dir)
    path = os.path.join(log_dir, "diff_mask_matrix_values.log")
    data = tensor.detach().cpu().float().numpy()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {name}\n")
        if extra_info:
            f.write(f"{extra_info}\n")
        f.write(np.array2string(data, precision=6, separator=", "))
        f.write("\n\n")


def generate_diff_sparse_mask(
    tensor: torch.Tensor,
    B_h: int,
    B_w: int,
    alpha: float,
    *,
    log_enabled: bool = False,
    log_dir: str = "./log",
) -> torch.Tensor:
    """
    基于层次差分逻辑生成稀疏掩码。

    Args:
        tensor: 输入张量，形状为 [Batch, B_hw, B_hw]。
        B_h: 行向量划分的段数。
        B_w: 每段的长度。
        alpha: 阈值步长。

    Returns:
        shape 同输入的 bool 掩码张量。
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

    _log_run_summary(
        log_enabled,
        log_dir,
        batch=batch_size,
        B_h=B_h,
        B_w=B_w,
        alpha=alpha,
    )

    values = tensor.to(dtype=torch.float32)
    segments = values.reshape(batch_size, num_rows, B_h, B_w)
    seg_max = segments.max(dim=-1).values
    row_max = seg_max.max(dim=-1).values
    residue = row_max.unsqueeze(-1) - seg_max

    _maybe_log_matrix(log_enabled, log_dir, "seg_max", seg_max)
    _maybe_log_matrix(log_enabled, log_dir, "row_max", row_max)
    _maybe_log_matrix(log_enabled, log_dir, "residue", residue)

    arange_k = torch.arange(1, 5, device=values.device, dtype=values.dtype)
    thresholds = seg_max.unsqueeze(-1) - float(alpha) * arange_k.view(1, 1, 1, 4)
    masks_candidate = segments.unsqueeze(-2) >= thresholds.unsqueeze(-1)

    selection = torch.zeros_like(residue, dtype=torch.long)
    selection = torch.where(residue <= alpha, torch.full_like(selection, 3), selection)
    selection = torch.where((residue > alpha) & (residue <= 2 * alpha), torch.full_like(selection, 2), selection)
    selection = torch.where((residue > 2 * alpha) & (residue <= 3 * alpha), torch.full_like(selection, 1), selection)
    zero_mask = residue > (4 * alpha)

    _maybe_log_matrix(log_enabled, log_dir, "selection", selection.float())
    _maybe_log_matrix(log_enabled, log_dir, "zero_mask", zero_mask.float())

    gather_index = selection.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, B_w)
    selected = torch.gather(masks_candidate, dim=3, index=gather_index).squeeze(3)
    selected = selected & (~zero_mask.unsqueeze(-1))

    result = selected.reshape(batch_size, num_rows, num_cols).to(dtype=torch.bool)
    _maybe_log_matrix(log_enabled, log_dir, "result_mask", result.float())
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    B_h, B_w = 2, 3
    alpha = 0.05
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
    mask = generate_diff_sparse_mask(sample, B_h=B_h, B_w=B_w, alpha=alpha, log_enabled=True, log_dir="./log")
    print("输入：", sample)
    print("掩码：", mask.to(dtype=torch.uint8))