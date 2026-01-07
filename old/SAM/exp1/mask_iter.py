import os
import time
import numpy as np
import torch
from typing import Callable, List, Tuple


def _normalize_weights(weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weights = torch.clamp(weights, min=eps)
    return weights / weights.sum()


def _ensure_log_dir(log_dir: str) -> None:
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def _log_run_summary(
    log_enabled: bool,
    log_dir: str,
    *,
    num_past_step: int,
    B_hw: int,
    num_block_sqrt: int,
    timesteps_past: List[int],
    timestep_now: int,
    alpha: Tuple[float, float, float],
    threshold_inter: float,
    d_disappear: int,
) -> None:
    if not log_enabled:
        return
    _ensure_log_dir(log_dir)
    summary_path = os.path.join(log_dir, "run_summary.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            f"[{timestamp}] predict_sparse_attn_mask call:\n"
            f"  num_past_step = {num_past_step}\n"
            f"  B_hw = {B_hw}\n"
            f"  num_block_sqrt = {num_block_sqrt}\n"
            f"  timesteps_past = {timesteps_past}\n"
            f"  timestep_now = {timestep_now}\n"
            f"  alpha = {alpha}\n"
            f"  threshold_inter = {threshold_inter}\n"
            f"  d_disappear = {d_disappear}\n\n"
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
    path = os.path.join(log_dir, "matrix_values.log")
    data = tensor.detach().cpu().float().numpy()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {name}\n")
        if extra_info:
            f.write(f"{extra_info}\n")
        f.write(np.array2string(data, precision=6, separator=", "))
        f.write("\n\n")


def predict_sparse_attn_mask(
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
    log_enabled: bool = False,
    log_dir: str = "./log",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        B_hw: 每个大块的边长。
        num_block_sqrt: 注意力矩阵块划分后的行/列块数。
        mask_past: 历史对角线块缓存 [num_past_step, num_block_sqrt, B_hw, B_hw]。
        timesteps_past: 历史时间戳。
        timestep_now: 当前时间戳。
        alpha: (alpha_w, alpha_h, alpha_t)。
        threshold_inter: 非对角块阈值。
        mask_diag_now: 当前时间步的对角线块掩码 [num_block_sqrt, B_hw, B_hw]。
        d_disappear: 非对角块与对角线的最大保留距离。
        num_past_step: 历史对角线块数量，默认 2。
    """
    assert 1 <= d_disappear <= num_block_sqrt - 1, "d_disappear 超出允许范围"
    assert len(timesteps_past) == num_past_step, "timesteps_past 长度需等于 num_past_step"
    assert mask_past.shape == (num_past_step, num_block_sqrt, B_hw, B_hw), \
        f"mask_past 形状不符，期望 {(num_past_step, num_block_sqrt, B_hw, B_hw)}"
    assert mask_diag_now.shape == (num_block_sqrt, B_hw, B_hw), \
        f"mask_diag_now 形状不符，期望 {(num_block_sqrt, B_hw, B_hw)}"
    _log_run_summary(
        log_enabled,
        log_dir,
        num_past_step=num_past_step,
        B_hw=B_hw,
        num_block_sqrt=num_block_sqrt,
        timesteps_past=timesteps_past,
        timestep_now=timestep_now,
        alpha=alpha,
        threshold_inter=threshold_inter,
        d_disappear=d_disappear,
    )

    device = mask_past.device
    mask_diag_now = mask_diag_now.to(device=device, dtype=torch.uint8)
    alpha_w, alpha_h, alpha_t = alpha

    L = B_hw * num_block_sqrt
    full_mask = torch.zeros(L, L, device=device, dtype=torch.uint8)

    for idx in range(num_block_sqrt):
        _maybe_log_matrix(
            log_enabled,
            log_dir,
            f"diag_block_{idx}",
            mask_diag_now[idx],
            extra_info=f"diagonal block index={idx}",
        )
        start = idx * B_hw
        full_mask[start:start + B_hw, start:start + B_hw] = mask_diag_now[idx]

    for qi in range(num_block_sqrt):
        for kj in range(num_block_sqrt):
            if qi == kj or abs(qi - kj) > d_disappear:
                continue

            mats = []
            weights = []

            spatial_dist = abs(qi - kj)
            d_h_row = dist_func(torch.tensor([0.0], device=device))[0].clamp_min(1e-6)
            d_w_row = dist_func(torch.tensor([float(spatial_dist)], device=device))[0].clamp_min(1e-6)
            d_h_col = dist_func(torch.tensor([float(spatial_dist)], device=device))[0].clamp_min(1e-6)
            d_w_col = dist_func(torch.tensor([0.0], device=device))[0].clamp_min(1e-6)

            for step in range(num_past_step):
                d_t_step = dist_func(
                    torch.tensor([float(abs(timestep_now - timesteps_past[step]))], device=device)
                )[0].clamp_min(1e-6)

                mat_row = mask_past[step, qi].float()
                weight_row = 1.0 / (alpha_h * d_h_row + alpha_w * d_w_row + alpha_t * d_t_step + 1e-6)
                mats.append(mat_row)
                weights.append(weight_row)

                mat_col = mask_past[step, kj].float()
                weight_col = 1.0 / (alpha_h * d_h_col + alpha_w * d_w_col + alpha_t * d_t_step + 1e-6)
                mats.append(mat_col)
                weights.append(weight_col)

            weights_block = _normalize_weights(torch.tensor(weights, device=device))
            block_stack = torch.stack(mats, dim=0)
            weighted_block = (weights_block.view(-1, 1, 1) * block_stack).sum(dim=0)

            _maybe_log_matrix(
                log_enabled,
                log_dir,
                f"inter_block_{qi}_{kj}",
                weighted_block,
                extra_info=f"off-diagonal block (row={qi}, col={kj}), threshold_inter={threshold_inter}",
            )

            block_mask = (weighted_block > threshold_inter).to(torch.uint8)

            q_start = qi * B_hw
            k_start = kj * B_hw
            full_mask[q_start:q_start + B_hw, k_start:k_start + B_hw] = block_mask

    mask_now = torch.cat([mask_past[1:], mask_diag_now.unsqueeze(0)], dim=0)
    return full_mask, mask_now


if __name__ == "__main__":
    num_past = 2
    B_hw = 2
    num_blocks = 3
    timesteps_hist = [999, 997]
    timestep_cur = 980
    alpha = (1.0, 1.0, 0.2)
    thr_inter = 0.6
    thr_intra = 0.4

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

    full_mask, mask_next = predict_sparse_attn_mask(
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
        log_enabled=True,
        log_dir="./log",
    )

    print("Full mask shape:", full_mask.shape)
    print(full_mask)
    print("\nNext history shape:", mask_next.shape)
    print(mask_next)