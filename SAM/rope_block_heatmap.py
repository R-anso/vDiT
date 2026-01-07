import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


_SUPPORTED_EXTS: Tuple[str, ...] = (".pt", ".pth")
_DIM_NAMES: Tuple[str, ...] = ("w", "h", "f")
_COMPONENT_KEYS: Tuple[str, ...] = ("A_w", "A_h", "A_f")
_COMPONENT_ALIASES: Dict[str, str] = {
    "a_w": "A_w",
    "aw": "A_w",
    "a_h": "A_h",
    "ah": "A_h",
    "a_f": "A_f",
    "af": "A_f",
}


def _prepare_output_path(
    input_file: str,
    output: Optional[str],
    suffix: str,
    root_dir: Optional[str] = None,
) -> str:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if not output:
        out_dir = os.path.dirname(input_file)
        return os.path.join(out_dir, f"{base_name}_{suffix}.png")

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
        return os.path.join(out_dir, f"{base_name}_{suffix}.png")

    root, ext = os.path.splitext(output)
    ext = ext or ".png"
    return f"{root}_{base_name}_{suffix}{ext}"


def _build_sort_key(path: str, pattern: re.Pattern | None) -> Tuple:
    base = os.path.basename(path)
    
    # 强制提取 It, L, H
    it_match = re.search(r"It(\d+)", base)
    l_match = re.search(r"L(\d+)", base)
    h_match = re.search(r"H(\d+)", base)
    
    if it_match and l_match and h_match:
        return (
            int(h_match.group(1)), 
            int(l_match.group(1)), 
            int(it_match.group(1)), 
            base
        )

    # 如果找不到标准的 It/L/H 标记，则回退到基于正则分组的排序
    if pattern:
        match = pattern.search(base)
        if match:
            parts: list = []
            for group in match.groups():
                try:
                    parts.append(int(group))
                except (TypeError, ValueError):
                    parts.append(group)
            parts.append(base)
            return tuple(parts)
    return (float("inf"), base)


def _collect_files(path: str, regex: Optional[str]) -> List[str]:
    """Collect candidate files from a path, mirroring attn_plot.py semantics."""
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
    """Load A_w/A_h/A_f tensors from a .pt/.pth file, similar to 3D_cluster_pred IO."""
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


def _quantize_blockwise(block_grid: torch.Tensor) -> torch.Tensor:
    block_min = block_grid.amin(dim=(-2, -1), keepdim=True)
    block_max = block_grid.amax(dim=(-2, -1), keepdim=True)
    denom = block_max - block_min
    scale = torch.where(denom > 0, denom / 255.0, torch.ones_like(denom))
    quantized = torch.where(
        denom > 0,
        ((block_grid - block_min) / scale - 128.0).round(),
        torch.zeros_like(block_grid),
    )
    return torch.clamp(quantized, -128, 127)


def _reshape_component(
    matrix: torch.Tensor,
    dims_order: Tuple[str, str, str],
    sizes: Dict[str, int],
    flatten: bool,
    normalize: bool,
) -> np.ndarray:
    """Reshape an attention component for block or flattened visualization."""
    l_w, l_h, l_f = sizes["w"], sizes["h"], sizes["f"]
    expected = l_w * l_h * l_f
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != expected:
        raise ValueError(
            "Attention component must be a square matrix with side length l_w * l_h * l_f."
        )

    attn6d = matrix.reshape(l_w, l_h, l_f, l_w, l_h, l_f)
    index_map = {
        "w_q": 0, "h_q": 1, "f_q": 2,
        "w_k": 3, "h_k": 4, "f_k": 5,
    }

    primary_dim = dims_order[0]          # 维度名："w"/"h"/"f"
    inner_dims = (dims_order[1], dims_order[2])
    primary_size = sizes[primary_dim]    # 维度大小：整数
    other1 = sizes[dims_order[1]]
    other2 = sizes[dims_order[2]]
    block_side = other1 * other2

    # 先内维(H/W)，后主维(primary)，用于轴置换
    q_axes = [index_map[f"{dim}_q"] for dim in (*inner_dims, primary_dim)]
    k_axes = [index_map[f"{dim}_k"] for dim in (*inner_dims, primary_dim)]
    permuted = attn6d.permute(q_axes + k_axes)

    # 构建块网格，块内为 l_primary × l_primary
    block_grid = permuted.reshape(block_side, primary_size, block_side, primary_size) \
                         .permute(0, 2, 1, 3)
    if normalize:
        block_grid = _quantize_blockwise(block_grid)

    if flatten:
        flat_blocks = block_grid.reshape(primary_size * primary_size, block_side * block_side)
        flat_matrix = flat_blocks.transpose(1, 0)
        return flat_matrix.detach().cpu().numpy()

    block_matrix = block_grid.permute(0, 2, 1, 3) \
                             .reshape(block_side * primary_size, block_side * primary_size)
    return block_matrix.detach().cpu().numpy()


def _plot_single_component(
    component: torch.Tensor,
    comp_key: str,
    sizes: Dict[str, int],
    output_path: str,
    cmap: str,
    title_prefix: str,
    dpi: int,
    flatten: bool,
    normalize: bool,
) -> None:
    orders = {
        "A_w": ("w", "h", "f"),
        "A_h": ("h", "w", "f"),
        "A_f": ("f", "w", "h"),
    }

    data2d = _reshape_component(
        component,
        orders[comp_key],
        sizes,
        flatten,
        normalize,
    )
    height, width = data2d.shape
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data2d,
        ax=ax,
        cmap=cmap,
        square=False,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(f"{title_prefix}{comp_key}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _parse_sizes(args: argparse.Namespace) -> Dict[str, int]:
    return {"w": args.l_w, "h": args.l_h, "f": args.l_f}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize block periodicity of A_w/A_h/A_f components with RoPE-aware reshaping."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a .pt/.pth file or a directory containing such files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory or file path. Defaults to alongside the input file.",
    )
    parser.add_argument(
        "--file_regex",
        type=str,
        default=r"score_It(\d+)_L(\d+)_H(\d+)",
        help="Optional regex to filter files when input is a directory.",
    )
    parser.add_argument("--l_w", type=int, required=True, help="Spatial latent width (W).")
    parser.add_argument("--l_h", type=int, required=True, help="Spatial latent height (H).")
    parser.add_argument("--l_f", type=int, required=True, help="Temporal latent length (F).")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading tensors (default: cpu).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Colormap for the heatmaps.",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="",
        help="Optional prefix for subplot titles.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for the output figure (default: 150).",
    )
    parser.add_argument(
        "--is_flatten",
        action="store_true",
        help="Use flattened view (blocks as columns, elements as rows).",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Apply per-block 8-bit asymmetric quantization (min->-128, max->127) before plotting.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sizes = _parse_sizes(args)

    files = _collect_files(args.input, args.file_regex)
    if not files:
        raise FileNotFoundError("No .pt/.pth files found for the given input.")

    input_root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    
    if args.output and (args.output.endswith(os.sep) or os.path.isdir(args.output) or not os.path.splitext(args.output)[1]):
        os.makedirs(args.output.rstrip(os.sep), exist_ok=True)

    suffix_map = {"A_w": "_w", "A_h": "_h", "A_f": "_f"}

    for file_path in files:
        components = _load_components(file_path, device)
        for comp_key in ("A_w", "A_h", "A_f"):
            suffix = f"rope_heatmap{suffix_map[comp_key]}"
            output_path = _prepare_output_path(
                file_path,
                args.output,
                suffix,
                root_dir=input_root,
            )
            _plot_single_component(
                components[comp_key],
                comp_key,
                sizes,
                output_path,
                args.cmap,
                args.title_prefix,
                args.dpi,
                args.is_flatten,
                args.norm,
            )
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()