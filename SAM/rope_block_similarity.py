import argparse
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as ssim  # optional
except ImportError:
    ssim = None


_SUPPORTED_EXTS: Tuple[str, ...] = (".pt", ".pth")
_COMPONENT_KEYS: Tuple[str, ...] = ("A_w", "A_h", "A_f")
_COMPONENT_ALIASES: Dict[str, str] = {
    "a_w": "A_w",
    "aw": "A_w",
    "a_h": "A_h",
    "ah": "A_h",
    "a_f": "A_f",
    "af": "A_f",
}
_COMPONENT_PRIMARY_DIM: Dict[str, str] = {
    "A_w": "w",
    "A_h": "h",
    "A_f": "f",
}
_DIM_NAMES: Tuple[str, ...] = ("w", "h", "f")


def _prepare_output_path(
    input_file: str,
    output: Optional[str],
    suffix: str,
    root_dir: Optional[str] = None,
) -> str:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if not output:
        out_dir = os.path.dirname(input_file)
        return os.path.join(out_dir, f"{base_name}_{suffix}")

    is_dir_hint = output.endswith(os.sep) or os.path.isdir(output) or not os.path.splitext(output)[1]
    if is_dir_hint:
        output_root = output.rstrip(os.sep)
        rel_dir = ""
        if root_dir:
            try:
                rel_path = os.path.relpath(os.path.dirname(input_file), root_dir)
                if rel_path != ".":
                    rel_dir = rel_path
            except ValueError:
                pass
        out_dir = os.path.join(output_root, rel_dir) if rel_dir else output_root
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{base_name}_{suffix}")

    root, _ = os.path.splitext(output)
    return f"{root}_{base_name}_{suffix}"


def _build_sort_key(path: str, pattern: re.Pattern | None) -> Tuple:
    base = os.path.basename(path)

    it_match = re.search(r"It(\d+)", base)
    l_match = re.search(r"L(\d+)", base)
    h_match = re.search(r"H(\d+)", base)

    if it_match and l_match and h_match:
        return (
            int(h_match.group(1)),
            int(l_match.group(1)),
            int(it_match.group(1)),
            base,
        )

    if pattern:
        match = pattern.search(base)
        if match:
            parts: List = []
            for group in match.groups():
                try:
                    parts.append(int(group))
                except (TypeError, ValueError):
                    parts.append(group)
            parts.append(base)
            return tuple(parts)
    return (float("inf"), base)


def _collect_files(path: str, regex: Optional[str]) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.lower().endswith(_SUPPORTED_EXTS) else []

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input path not found: {path}")

    pattern = re.compile(regex) if regex else None
    collected: List[str] = []
    for root, _, files in os.walk(path):
        for fname in files:
            if not fname.lower().endswith(_SUPPORTED_EXTS):
                continue
            if pattern and not pattern.search(fname):
                continue
            collected.append(os.path.join(root, fname))
    collected.sort(key=lambda p: _build_sort_key(p, pattern))
    return collected


def _load_components(file_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    raw = torch.load(file_path, map_location=device)

    if isinstance(raw, dict):
        components: Dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized = _COMPONENT_ALIASES.get(key.lower(), key)
            if normalized in _COMPONENT_KEYS:
                components[normalized] = value.detach().float()
        missing = [k for k in _COMPONENT_KEYS if k not in components]
        if missing:
            raise ValueError(f"{file_path} missing components: {missing}")
        return components

    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        tensors = [torch.as_tensor(item, device=device).float() for item in raw]
        return {"A_w": tensors[0], "A_h": tensors[1], "A_f": tensors[2]}

    if isinstance(raw, torch.Tensor) and raw.ndim == 3 and raw.shape[0] == 3:
        return {
            "A_w": raw[0].detach().to(device=device, dtype=torch.float32),
            "A_h": raw[1].detach().to(device=device, dtype=torch.float32),
            "A_f": raw[2].detach().to(device=device, dtype=torch.float32),
        }

    raise TypeError(
        f"Unsupported data structure in {file_path}. Expected dict with A_w/A_h/A_f or a length-3 list/tuple/tensor."
    )


def _extract_block_grid(
    matrix: torch.Tensor,
    primary_dim: str,
    sizes: Dict[str, int],
) -> np.ndarray:
    # 强制在 CPU 上做重排和视图操作，避免在 CUDA 上产生巨大中间张量导致 OOM
    matrix = matrix.detach().to("cpu", non_blocking=False)

    l_w, l_h, l_f = sizes["w"], sizes["h"], sizes["f"]
    expected = l_w * l_h * l_f
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != expected:
        raise ValueError("Attention component must be a square matrix with side length l_w * l_h * l_f.")

    attn6d = matrix.reshape(l_f, l_h, l_w, l_f, l_h, l_w)
    index_map = {
        "f_q": 0,
        "h_q": 1,
        "w_q": 2,
        "f_k": 3,
        "h_k": 4,
        "w_k": 5,
    }

    other_dims = [dim for dim in _DIM_NAMES if dim != primary_dim]
    q_axes = [index_map[f"{primary_dim}_q"]] + [index_map[f"{dim}_q"] for dim in other_dims]
    k_axes = [index_map[f"{primary_dim}_k"]] + [index_map[f"{dim}_k"] for dim in other_dims]

    permuted = attn6d.permute(q_axes + k_axes).reshape(
        sizes[primary_dim],
        sizes[other_dims[0]] * sizes[other_dims[1]],
        sizes[primary_dim],
        sizes[other_dims[0]] * sizes[other_dims[1]],
    )
    grid = permuted.permute(0, 2, 1, 3)
    return grid.detach().cpu().numpy()


def _flatten_blocks(block_grid: np.ndarray) -> np.ndarray:
    return block_grid.reshape(-1, block_grid.shape[-1] * block_grid.shape[-2])


def _flatten_positions(block_grid: np.ndarray) -> np.ndarray:
    return _flatten_blocks(block_grid).T


def _cosine_similarity(vec: np.ndarray, ref: np.ndarray) -> float:
    denom = np.linalg.norm(vec) * np.linalg.norm(ref)
    if denom == 0:
        return float("nan")
    return float(np.dot(vec, ref) / denom)


def _pearson_correlation(vec: np.ndarray, ref: np.ndarray) -> float:
    vec_c = vec - vec.mean()
    ref_c = ref - ref.mean()
    denom = np.linalg.norm(vec_c) * np.linalg.norm(ref_c)
    if denom == 0:
        return float("nan")
    return float(np.dot(vec_c, ref_c) / denom)


def _compute_center_consistency(
    vectors: np.ndarray,
    enable_ssim: bool,
    ssim_side: Optional[int],
) -> Dict[str, float]:
    centroid = vectors.mean(axis=0)
    metrics: Dict[str, float] = {}
    cos_scores: List[float] = []
    pearson_scores: List[float] = []
    ssim_scores: List[float] = []
    centroid_norm = np.linalg.norm(centroid)
    centroid_c = centroid - centroid.mean()
    centroid_c_norm = np.linalg.norm(centroid_c)
    can_do_ssim = enable_ssim and ssim is not None and ssim_side is not None and ssim_side * ssim_side == centroid.shape[0]
    for vec in vectors:
        cos_scores.append(_cosine_similarity(vec, centroid))
        pearson_scores.append(_pearson_correlation(vec, centroid))
        if can_do_ssim:
            centroid_block = centroid.reshape(ssim_side, ssim_side)
            vec_block = vec.reshape(ssim_side, ssim_side)
            data_range = float(max(vec_block.max(), centroid_block.max()) - min(vec_block.min(), centroid_block.min()))
            if data_range == 0:
                ssim_scores.append(1.0 if np.allclose(vec_block, centroid_block) else 0.0)
            else:
                ssim_scores.append(float(ssim(vec_block, centroid_block, data_range=data_range)))
    metrics["cosine_mean"] = float(np.nanmean(cos_scores))
    metrics["cosine_std"] = float(np.nanstd(cos_scores))
    metrics["pearson_mean"] = float(np.nanmean(pearson_scores))
    metrics["pearson_std"] = float(np.nanstd(pearson_scores))
    if enable_ssim:
        if can_do_ssim:
            metrics["ssim_mean"] = float(np.nanmean(ssim_scores))
            metrics["ssim_std"] = float(np.nanstd(ssim_scores))
        else:
            metrics["ssim_mean"] = float("nan")
            metrics["ssim_std"] = float("nan")
    metrics["centroid_norm"] = float(centroid_norm)
    metrics["centroid_centered_norm"] = float(centroid_c_norm)
    return metrics


def _compute_svd_metrics(vectors: np.ndarray) -> Dict[str, float]:
    centered = vectors - vectors.mean(axis=0, keepdims=True)

    if np.allclose(centered, 0):
        return {
            "singular_values": "all_zero",
            "dominant_variance_ratio": float("nan"),
            "effective_rank": 0,
        }

    try:
        singular_values = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "singular_values": "svd_failed",
            "dominant_variance_ratio": float("nan"),
            "effective_rank": float("nan"),
        }

    variances = singular_values ** 2
    total_variance = float(variances.sum())
    dominant_ratio = float(variances[0] / total_variance) if total_variance > 0 else float("nan")

    normalized = variances / total_variance if total_variance > 0 else variances
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(normalized * np.log(normalized + 1e-12))
    effective_rank = float(np.exp(entropy))

    return {
        "singular_values": ", ".join(f"{sv:.4e}" for sv in singular_values[:8]),
        "dominant_variance_ratio": dominant_ratio,
        "effective_rank": effective_rank,
    }


def _str_to_torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16" or name == "fp16" or name == "half":
        return torch.float16
    if name == "bfloat16" or name == "bf16":
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def _compute_ssim_mean_std(vectors: np.ndarray, ssim_side: Optional[int]) -> Tuple[float, float]:
    if ssim is None or ssim_side is None or ssim_side * ssim_side != vectors.shape[1]:
        return float("nan"), float("nan")
    centroid = vectors.mean(axis=0)
    centroid_block = centroid.reshape(ssim_side, ssim_side)
    scores: List[float] = []
    for vec in vectors:
        vb = vec.reshape(ssim_side, ssim_side)
        data_range = float(max(vb.max(), centroid_block.max()) - min(vb.min(), centroid_block.min()))
        if data_range == 0:
            scores.append(1.0 if np.allclose(vb, centroid_block) else 0.0)
        else:
            scores.append(float(ssim(vb, centroid_block, data_range=data_range)))
    return float(np.nanmean(scores)), float(np.nanstd(scores))


@torch.no_grad()
def _compute_center_consistency_gpu(
    vectors_np: np.ndarray,
    enable_ssim: bool,
    ssim_side: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    row_chunk: int,
) -> Dict[str, float]:
    # 先在 CPU 上计算质心，避免把整矩阵搬上 GPU
    centroid_np = vectors_np.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid_np))
    centroid_c_np = centroid_np - centroid_np.mean()
    centroid_c_norm = float(np.linalg.norm(centroid_c_np))

    # 辅助：统计均值/方差（忽略 NaN/Inf）
    def init_stats():
        return {"count": 0, "sum": 0.0, "sumsq": 0.0}

    def update_stats(stats, scores: torch.Tensor):
        if scores.numel() == 0:
            return
        valid = torch.isfinite(scores)
        if valid.any():
            s = scores[valid]
            stats["count"] += int(s.numel())
            stats["sum"] += float(s.sum().item())
            stats["sumsq"] += float((s * s).sum().item())

    def finalize_stats(stats) -> Tuple[float, float]:
        n = stats["count"]
        if n == 0:
            return float("nan"), float("nan")
        mean = stats["sum"] / n
        var = max(stats["sumsq"] / n - mean * mean, 0.0)
        return mean, float(np.sqrt(var))

    # 将质心搬到 GPU（低精度可选），计算时用 fp32 做累计以保证数值稳定
    c_gpu = torch.from_numpy(centroid_np).to(device=device, dtype=dtype, non_blocking=False)
    c_c_gpu = torch.from_numpy(centroid_c_np).to(device=device, dtype=dtype, non_blocking=False)
    c_norm = torch.norm(c_gpu.to(torch.float32))
    c_c_norm = torch.norm(c_c_gpu.to(torch.float32))

    cos_stats = init_stats()
    pearson_stats = init_stats()

    # 边界：质心为零向量时，cos/pearson 均不可定义
    if c_norm.item() == 0.0:
        cos_mean, cos_std = float("nan"), float("nan")
    if c_c_norm.item() == 0.0:
        p_mean, p_std = float("nan"), float("nan")

    # 分批把行（样本）搬上 GPU 进行点积与归一化，显存占用 ~ O(row_chunk + len(centroid))
    num_rows = vectors_np.shape[0]
    for start in range(0, num_rows, row_chunk):
        end = min(start + row_chunk, num_rows)
        batch_np = vectors_np[start:end]  # (B, D)
        b_gpu = torch.from_numpy(batch_np).to(device=device, dtype=dtype, non_blocking=False)

        # cosine
        if c_norm.item() > 0.0:
            numer = (b_gpu.to(torch.float32) * c_gpu.to(torch.float32)).sum(dim=1)  # (B,)
            denom = torch.norm(b_gpu.to(torch.float32), dim=1) * c_norm  # (B,)
            cos_scores = numer / (denom + 1e-12)
            update_stats(cos_stats, cos_scores)

        # pearson
        if c_c_norm.item() > 0.0:
            b_mean = b_gpu.mean(dim=1, keepdim=True)
            b_c = b_gpu - b_mean
            numer_p = (b_c.to(torch.float32) * c_c_gpu.to(torch.float32)).sum(dim=1)
            denom_p = torch.norm(b_c.to(torch.float32), dim=1) * c_c_norm
            p_scores = numer_p / (denom_p + 1e-12)
            update_stats(pearson_stats, p_scores)

        # 及时释放 batch 显存
        del b_gpu

    cos_mean, cos_std = finalize_stats(cos_stats)
    p_mean, p_std = finalize_stats(pearson_stats)

    # 可选：SSIM 仍在 CPU 上计算（块数通常不大，稳定省显存）
    ssim_mean, ssim_std = float("nan"), float("nan")
    if enable_ssim and ssim is not None:
        ssim_mean, ssim_std = _compute_ssim_mean_std(vectors_np, ssim_side)

    return {
        "cosine_mean": cos_mean,
        "cosine_std": cos_std,
        "pearson_mean": p_mean,
        "pearson_std": p_std,
        "centroid_norm": centroid_norm,
        "centroid_centered_norm": centroid_c_norm,
        "ssim_mean": ssim_mean if enable_ssim else float("nan"),
        "ssim_std": ssim_std if enable_ssim else float("nan"),
    }


def _setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("rope_block_similarity")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.addHandler(console_handler)

    logger.debug("Logger initialised.")
    return logger


def _determine_log_path(args: argparse.Namespace, input_root: str) -> str:
    if args.log_path:
        log_dir = os.path.dirname(args.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        return args.log_path

    if args.output:
        base = _prepare_output_path(input_root, args.output, "similarity_log", root_dir=input_root)
        return f"{base}.log"

    default_dir = os.path.dirname(args.input) if os.path.isfile(args.input) else args.input
    os.makedirs(default_dir, exist_ok=True)
    return os.path.join(default_dir, "rope_block_similarity.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate similarity of RoPE-decomposed attention blocks via centroid consistency and SVD. Low-VRAM + optional GPU streaming."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to a .pt/.pth file or directory.")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file hint (for log placement).")
    parser.add_argument("--file_regex", type=str, default=r"score_It(\d+)_L(\d+)_H(\d+)", help="Regex filter for filenames.")
    parser.add_argument("--l_w", type=int, required=True, help="Spatial latent width (W).")
    parser.add_argument("--l_h", type=int, required=True, help="Spatial latent height (H).")
    parser.add_argument("--l_f", type=int, required=True, help="Temporal latent length (F).")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading tensors (default: cpu).")
    parser.add_argument(
        "--components",
        type=str,
        default="A_w,A_h,A_f",
        help="Comma-separated list of components to analyse (subset of A_w,A_h,A_f).",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Explicit log file path.")
    parser.add_argument(
        "--enable_ssim",
        action="store_true",
        help="Compute SSIM against centroid (requires scikit-image).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging.")

    # 新增：低显存 GPU 加速选项（cosine/pearson 流式在 GPU 上计算）
    parser.add_argument(
        "--metrics_on_gpu",
        action="store_true",
        help="Compute cosine/pearson on GPU with low VRAM streaming.",
    )
    parser.add_argument(
        "--gpu_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="GPU dtype for metrics streaming (default: float16).",
    )
    parser.add_argument(
        "--gpu_row_chunk",
        type=int,
        default=8192,
        help="Row batch size when streaming metrics to GPU (default: 8192).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_device = torch.device(args.device)
    sizes = {"w": args.l_w, "h": args.l_h, "f": args.l_f}

    files = _collect_files(args.input, args.file_regex)
    if not files:
        raise FileNotFoundError("No .pt/.pth files found for the given input.")

    input_root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    log_path = _determine_log_path(args, input_root)
    logger = _setup_logger(log_path, args.verbose)

    requested_components = [comp.strip() for comp in args.components.split(",") if comp.strip()]
    for comp in requested_components:
        if comp not in _COMPONENT_KEYS:
            raise ValueError(f"Unsupported component requested: {comp}")

    if args.enable_ssim and ssim is None:
        logger.warning("SSIM computation requested but scikit-image is not available. SSIM scores will be NaN.")

    # 决定 metrics 计算所用设备（流式 GPU）
    metrics_device = torch.device("cuda") if (args.metrics_on_gpu and torch.cuda.is_available()) else torch.device("cpu")
    gpu_dtype = _str_to_torch_dtype(args.gpu_dtype)
    if args.metrics_on_gpu and metrics_device.type != "cuda":
        logger.warning("metrics_on_gpu is set but CUDA is not available; falling back to CPU.")

    logger.info("Starting block similarity analysis.")
    logger.info("Configuration: %s", vars(args))
    if metrics_device.type == "cuda":
        logger.info("Metrics device: CUDA (dtype=%s, row_chunk=%d)", str(gpu_dtype).split('.')[-1], args.gpu_row_chunk)
    else:
        logger.info("Metrics device: CPU")

    for file_path in files:
        logger.info("Processing file: %s", file_path)
        try:
            components = _load_components(file_path, load_device)
        except Exception as exc:
            logger.error("Failed to load components from %s: %s", file_path, exc)
            continue

        for comp_key in requested_components:
            primary_dim = _COMPONENT_PRIMARY_DIM[comp_key]
            try:
                block_grid = _extract_block_grid(components[comp_key], primary_dim, sizes)
            except Exception as exc:
                logger.error("Failed to extract blocks for %s in %s: %s", comp_key, file_path, exc)
                continue

            num_blocks = block_grid.shape[0] * block_grid.shape[1]
            inner_size = block_grid.shape[-1]

            logger.info(
                "Component %s: primary=%s, blocks=%d, inner_shape=%dx%d",
                comp_key,
                primary_dim,
                num_blocks,
                inner_size,
                inner_size,
            )

            # 减显存：block_grid 是 numpy(cupy 未用)，后续按需分批搬到 GPU 计算
            vectors_blocks = _flatten_blocks(block_grid)            # 行：块，列：块内
            vectors_positions = _flatten_positions(block_grid)      # 行：位置，列：所有块

            # block-wise
            ssim_side_block = block_grid.shape[-1]
            if metrics_device.type == "cuda":
                center_block = _compute_center_consistency_gpu(
                    vectors_blocks,
                    enable_ssim=args.enable_ssim,
                    ssim_side=ssim_side_block,
                    device=metrics_device,
                    dtype=gpu_dtype,
                    row_chunk=args.gpu_row_chunk,
                )
            else:
                center_block = _compute_center_consistency(
                    vectors_blocks, enable_ssim=args.enable_ssim, ssim_side=ssim_side_block
                )
            svd_block = _compute_svd_metrics(vectors_blocks)
            logger.info(
                "  Block-wise -> cosine_mean=%.6f (std=%.6f), pearson_mean=%.6f (std=%.6f), centroid_norm=%.4e",
                center_block["cosine_mean"],
                center_block["cosine_std"],
                center_block["pearson_mean"],
                center_block["pearson_std"],
                center_block["centroid_norm"],
            )
            if args.enable_ssim:
                logger.info(
                    "  Block-wise SSIM -> mean=%s (std=%s)",
                    f"{center_block.get('ssim_mean', float('nan')):.6f}" if not np.isnan(center_block.get("ssim_mean", float("nan"))) else "NaN",
                    f"{center_block.get('ssim_std', float('nan')):.6f}" if not np.isnan(center_block.get("ssim_std", float("nan"))) else "NaN",
                )
            logger.info(
                "  Block-wise SVD -> dominant_variance_ratio=%s, effective_rank=%s, singular_values=%s",
                f"{svd_block['dominant_variance_ratio']:.6f}" if isinstance(svd_block["dominant_variance_ratio"], float) else svd_block["dominant_variance_ratio"],
                f"{svd_block['effective_rank']:.6f}" if isinstance(svd_block["effective_rank"], float) else svd_block["effective_rank"],
                svd_block["singular_values"],
            )

            # position-wise（行数可能很大，GPU 流式特别有用）
            pos_side = int(np.sqrt(vectors_positions.shape[1])) if int(np.sqrt(vectors_positions.shape[1])) ** 2 == vectors_positions.shape[1] else None
            if metrics_device.type == "cuda":
                center_pos = _compute_center_consistency_gpu(
                    vectors_positions,
                    enable_ssim=args.enable_ssim,
                    ssim_side=pos_side,
                    device=metrics_device,
                    dtype=gpu_dtype,
                    row_chunk=args.gpu_row_chunk,
                )
            else:
                center_pos = _compute_center_consistency(
                    vectors_positions, enable_ssim=args.enable_ssim, ssim_side=pos_side
                )
            svd_pos = _compute_svd_metrics(vectors_positions)
            logger.info(
                "  Position-wise -> cosine_mean=%.6f (std=%.6f), pearson_mean=%.6f (std=%.6f), centroid_norm=%.4e",
                center_pos["cosine_mean"],
                center_pos["cosine_std"],
                center_pos["pearson_mean"],
                center_pos["pearson_std"],
                center_pos["centroid_norm"],
            )
            if args.enable_ssim:
                logger.info(
                    "  Position-wise SSIM -> mean=%s (std=%s)",
                    f"{center_pos.get('ssim_mean', float('nan')):.6f}" if not np.isnan(center_pos.get('ssim_mean', float('nan'))) else "NaN",
                    f"{center_pos.get('ssim_std', float('nan')):.6f}" if not np.isnan(center_pos.get('ssim_std', float('nan'))) else "NaN",
                )
            logger.info(
                "  Position-wise SVD -> dominant_variance_ratio=%s, effective_rank=%s, singular_values=%s",
                f"{svd_pos['dominant_variance_ratio']:.6f}" if isinstance(svd_pos["dominant_variance_ratio"], float) else svd_pos["dominant_variance_ratio"],
                f"{svd_pos['effective_rank']:.6f}" if isinstance(svd_pos["effective_rank"], float) else svd_pos["effective_rank"],
                svd_pos["singular_values"],
            )

    logger.info("Analysis complete. Log stored at %s", log_path)


if __name__ == "__main__":
    main()