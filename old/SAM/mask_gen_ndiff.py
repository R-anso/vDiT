import os
import time
import numpy as np
import torch
import math


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
    beta: float,
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
            f"  alpha = {alpha}\n"
            f"  beta = {beta}\n\n"
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
    beta: float,
    *,
    log_enabled: bool = False,
    log_dir: str = "./log",
    mode: str = "ndiff",
) -> torch.Tensor:
    """
    基于层次差分逻辑生成稀疏掩码。
    ...
        alpha: 阈值步长。
        beta: 每级段保留比例系数，范围 [0, 0.25]。
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

    _log_run_summary(
        log_enabled,
        log_dir,
        batch=batch_size,
        B_h=B_h,
        B_w=B_w,
        alpha=alpha,
        beta=beta,
    )

    values = tensor.to(dtype=torch.float32)
    segments = values.reshape(batch_size, num_rows, B_h, B_w)

    if mode == "ndiff":
        segment_max = segments.max(dim=-1).values
        row_max = segment_max.max(dim=-1, keepdim=True).values
        diff = row_max - segment_max

        grades = torch.zeros_like(diff, dtype=torch.int64)
        grades = torch.where(diff <= alpha, torch.full_like(grades, 3), grades)
        grades = torch.where(
            (diff > alpha) & (diff <= 2 * alpha),
            torch.full_like(grades, 2),
            grades,
        )
        grades = torch.where(
            (diff > 2 * alpha) & (diff <= 3 * alpha),
            torch.full_like(grades, 1),
            grades,
        )

        keep_counts = torch.ceil(grades.to(torch.float32) * float(beta) * B_w).to(torch.int64)
        keep_counts = torch.clamp(keep_counts, max=B_w)

        order = segments.argsort(dim=-1, descending=True)
        ranks = torch.argsort(order, dim=-1)
        segment_mask = ranks < keep_counts.unsqueeze(-1)
        result = segment_mask.reshape(batch_size, num_rows, num_cols).to(dtype=torch.bool)
        _maybe_log_matrix(log_enabled, log_dir, "segment_max", segment_max)
        _maybe_log_matrix(log_enabled, log_dir, "row_max", row_max)
        _maybe_log_matrix(log_enabled, log_dir, "diff", diff)
        _maybe_log_matrix(log_enabled, log_dir, "grades", grades.to(torch.float32))
        _maybe_log_matrix(log_enabled, log_dir, "keep_counts", keep_counts.to(torch.float32))
    else:
        keep_elems = int(math.ceil(beta * num_cols))
        if keep_elems <= 0:
            result = torch.zeros_like(values, dtype=torch.bool)
        else:
            keep_elems = min(keep_elems, num_cols)
            topk_indices = torch.topk(values, k=keep_elems, dim=-1, largest=True).indices
            result = torch.zeros_like(values, dtype=torch.bool)
            result.scatter_(-1, topk_indices, True)

        if log_enabled:
            _maybe_log_matrix(log_enabled, log_dir, "rows_topk_values", values)
            _maybe_log_matrix(
                log_enabled,
                log_dir,
                "keep_elems_topk",
                torch.full((1,), float(keep_elems), device=values.device),
            )

    _maybe_log_matrix(log_enabled, log_dir, "result_mask", result.float())
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
    mask = generate_diff_sparse_mask(
        sample,
        B_h=B_h,
        B_w=B_w,
        alpha=alpha,
        beta=beta,
        log_enabled=True,
        log_dir="./log",
        mode='topk',
    )
    print("输入：", sample)
    print("掩码：", mask.to(dtype=torch.uint8))