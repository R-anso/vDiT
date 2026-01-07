import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

_SUPPORTED_EXTS: Tuple[str, ...] = (".pt", ".pth")
_COMPONENT_KEYS: Tuple[str, ...] = ("Q", "K", "Q_w", "Q_h", "Q_f", "K_w", "K_h", "K_f")
_COMPONENT_ALIASES: Dict[str, str] = {
    "q": "Q",
    "k": "K",
    "q_w": "Q_w", "qw": "Q_w",
    "q_h": "Q_h", "qh": "Q_h",
    "q_f": "Q_f", "qf": "Q_f",
    "k_w": "K_w", "kw": "K_w",
    "k_h": "K_h", "kh": "K_h",
    "k_f": "K_f", "kf": "K_f",
}


def _prepare_output_path(input_file: str, output: Optional[str], suffix: str, root_dir: Optional[str] = None) -> str:
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
    it_match = re.search(r"It(\d+)", base)
    l_match = re.search(r"L(\d+)", base)
    h_match = re.search(r"H(\d+)", base)
    if it_match and l_match and h_match:
        return (int(h_match.group(1)), int(l_match.group(1)), int(it_match.group(1)), base)
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
    components: Dict[str, torch.Tensor] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized = _COMPONENT_ALIASES.get(key.lower(), key)
            if normalized in _COMPONENT_KEYS:
                components[normalized] = value.detach().float()
    elif isinstance(raw, torch.Tensor) and raw.ndim == 3 and raw.shape[0] in (3, 6):
        # 兼容直接堆叠的张量：假设顺序为 [Q_w,Q_h,Q_f] 或 [Q_w,Q_h,Q_f,K_w,K_h,K_f]
        names = ("Q_w", "Q_h", "Q_f", "K_w", "K_h", "K_f")
        for i in range(min(raw.shape[0], len(names))):
            components[names[i]] = raw[i].detach().float()
    else:
        raise TypeError(f"Unsupported data structure in {file_path}. Expect dict with Q*/K* tensors or stacked tensor.")

    # [新增] 自动合成完整 Q/K (如果缺失但子组件齐全)
    for prefix in ("Q", "K"):
        if prefix not in components:
            f_key, h_key, w_key = f"{prefix}_f", f"{prefix}_h", f"{prefix}_w"
            if all(k in components for k in (f_key, h_key, w_key)):
                # 按照 f-h-w 顺序拼接，f 在低维度 (dim=-1 的起始部分)
                components[prefix] = torch.cat(
                    [components[f_key], components[h_key], components[w_key]], 
                    dim=-1
                )

    if not components:
        raise ValueError(f"{file_path} contains no Q/K components.")
    return components


def _quantize_per_token(array: np.ndarray) -> np.ndarray:
    # array: [L, D], symmetric per-token to int8
    max_abs = np.max(np.abs(array), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    scale = max_abs / 127.0
    q = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return q


def _build_grid(comp: torch.Tensor, comp_key: str, sizes: Dict[str, int], gap: int, quant: bool = True, gen_mask: bool = False) -> np.ndarray:
    l_w, l_h, l_f = sizes["w"], sizes["h"], sizes["f"]
    data = comp.detach().cpu().numpy()
    if data.ndim != 2:
        raise ValueError(f"{comp_key} tensor must be 2D [L, D], got {data.shape}")
    L, dim = data.shape
    expected = l_w * l_h * l_f
    if L != expected:
        raise ValueError(f"{comp_key} length mismatch: got {L}, expected {expected} (= l_w*l_h*l_f)")

    if gen_mask and comp_key in ("Q", "K"):
        # Mask generation based on absolute value ranking per token
        data_proc = np.zeros((L, dim), dtype=np.float32)
        d1 = dim // 4
        d2 = dim // 2
        
        abs_data = np.abs(data)
        idx = np.argsort(abs_data, axis=1)
        rows = np.arange(L)[:, None]
        
        # Top 25% absolute values -> Red (127)
        data_proc[rows, idx[:, -d1:]] = 127.0
        # Next 25% absolute values -> Blue (-127)
        data_proc[rows, idx[:, -d2:-d1]] = -127.0
    elif quant:
        data_proc = _quantize_per_token(data)  # int8
    else:
        data_proc = data  # float32

    data_proc = data_proc.reshape(l_w, l_h, l_f, dim)  # order: w, h, f

    if comp_key in ("Q", "K") or comp_key.endswith("_w"):
        block_rows = l_w
        grid_y, grid_x = l_f, l_h  # vertical=f, horizontal=h
        def _block(gx: int, gy: int) -> np.ndarray:
            h = gx
            f = gy
            return data_proc[:, h, f, :]
    elif comp_key.endswith("_h"):
        block_rows = l_h
        grid_y, grid_x = l_f, l_w  # vertical=f, horizontal=w
        def _block(gx: int, gy: int) -> np.ndarray:
            w = gx
            f = gy
            return data_proc[w, :, f, :]
    elif comp_key.endswith("_f"):
        block_rows = l_f
        grid_y, grid_x = l_h, l_w  # vertical=h, horizontal=w
        def _block(gx: int, gy: int) -> np.ndarray:
            w = gx
            h = gy
            return data_proc[w, h, :, :]
    else:
        raise ValueError(f"Unknown component key: {comp_key}")

    block_h, block_w = block_rows, dim
    canvas_h = grid_y * block_h + (grid_y - 1) * gap
    canvas_w = grid_x * block_w + (grid_x - 1) * gap
    canvas = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)

    for gy in range(grid_y):
        for gx in range(grid_x):
            blk = _block(gx, gy)
            r0 = gy * (block_h + gap)
            c0 = gx * (block_w + gap)
            canvas[r0:r0 + block_h, c0:c0 + block_w] = blk

    return canvas


def _plot_grid(grid: np.ndarray, out_path: str, title: str, cmap: str, dpi: int, quant: bool = True, force_range: bool = False) -> None:
    mask = np.isnan(grid)
    height, width = grid.shape
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize)
    
    # 如果不量化且不是生成掩码，则不固定 vmin/vmax，让 seaborn 自动根据数据范围缩放
    vmin, vmax = (-127, 127) if (quant or force_range) else (None, None)
    
    sns.heatmap(
        grid,
        ax=ax,
        cmap=cmap,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Q/K RoPE components with per-token 8bit symmetric quantization."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to a .pt/.pth file or directory.")
    parser.add_argument("--output", type=str, default=None, help="Output dir or file path. Defaults alongside input.")
    parser.add_argument(
        "--file_regex",
        type=str,
        default=r"(Q|K)_It(\d+)_L(\d+)_H(\d+)",
        help="Regex to filter files when input is a directory.",
    )
    parser.add_argument("--l_w", type=int, required=True, help="Spatial latent width (W).")
    parser.add_argument("--l_h", type=int, required=True, help="Spatial latent height (H).")
    parser.add_argument("--l_f", type=int, required=True, help="Temporal latent length (F).")
    parser.add_argument(
        "--components",
        type=str,
        default="Q,K,Q_w,Q_h,Q_f,K_w,K_h,K_f",
        help="Comma-separated list of components to plot (subset of Q,K,Q_w,Q_h,Q_f,K_w,K_h,K_f).",
    )
    parser.add_argument("--gap", type=int, default=1, help="Pixel gap between blocks.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading tensors.")
    parser.add_argument("--cmap", type=str, default="coolwarm", help="Colormap.")
    parser.add_argument("--title_prefix", type=str, default="", help="Optional prefix for titles.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output figure.")
    # 添加量化开关参数
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser.add_argument(
        "--quant", 
        type=str2bool, 
        nargs='?', 
        const=True, 
        default=True, 
        help="Enable/disable per-token 8bit symmetric quantization (default: True)."
    )
    parser.add_argument(
        "--gen_mask",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Generate a mask visualization for Q/K: first 25% red, second 25% blue."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sizes = {"w": args.l_w, "h": args.l_h, "f": args.l_f}
    want_components = [c.strip() for c in args.components.split(",") if c.strip()]
    want_components = [c if c in _COMPONENT_KEYS else _COMPONENT_ALIASES.get(c.lower(), c) for c in want_components]

    files = _collect_files(args.input, args.file_regex)
    if not files:
        raise FileNotFoundError("No .pt/.pth files found for the given input.")

    input_root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    if args.output and (args.output.endswith(os.sep) or os.path.isdir(args.output) or not os.path.splitext(args.output)[1]):
        os.makedirs(args.output.rstrip(os.sep), exist_ok=True)

    for file_path in files:
        components = _load_components(file_path, device)
        for comp_key in want_components:
            if comp_key not in components:
                continue
            
            is_mask_mode = args.gen_mask and comp_key in ("Q", "K")
            grid = _build_grid(components[comp_key], comp_key, sizes, args.gap, quant=args.quant, gen_mask=args.gen_mask)
            
            suffix = f"qk_heatmap_{comp_key}"
            if is_mask_mode:
                suffix += "_mask"
            elif not args.quant:
                suffix += "_noquant"
                
            out_path = _prepare_output_path(file_path, args.output, suffix, root_dir=input_root)
            title = f"{args.title_prefix}{comp_key}"
            if is_mask_mode:
                title += " (Mask)"
                
            _plot_grid(grid, out_path, title, args.cmap, args.dpi, quant=args.quant, force_range=is_mask_mode)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()