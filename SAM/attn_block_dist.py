import argparse
import json
import os
import re
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


SUPPORTED_EXTS = (".npy", ".json", ".txt", ".pt", ".pth")


def load_score_file(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return None

    try:
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            data = np.array(payload["scores"]) if isinstance(payload, dict) and "scores" in payload else np.array(payload)
        elif path.endswith(".txt"):
            data = np.loadtxt(path)
        elif path.endswith((".pt", ".pth")):
            payload = torch.load(path, map_location="cpu")
            data = payload.detach().cpu().numpy() if isinstance(payload, torch.Tensor) else np.asarray(payload)
        else:
            print(f"[Error] Unsupported file format: {path}")
            return None
        return data
    except Exception as exc:
        print(f"[Error] Could not load {path}: {exc}")
        return None


def collect_files(path: str, extensions: Iterable[str], regex: str | None) -> list[str]:
    files: list[str] = []
    if os.path.isfile(path):
        if path.lower().endswith(tuple(extensions)):
            files.append(path)
        return files

    pattern = re.compile(regex) if regex else None
    for root, _, names in os.walk(path):
        for name in names:
            if not name.lower().endswith(tuple(extensions)):
                continue
            if pattern and not pattern.search(name):
                continue
            files.append(os.path.join(root, name))
    return sorted(files)


def parse_indices(filename: str, pattern: re.Pattern[str]) -> tuple[int, int, int]:
    match = pattern.search(os.path.basename(filename))
    if not match:
        return (0, 0, 0)
    groups = match.groups()
    vals = [int(g) for g in groups[:3]]
    while len(vals) < 3:
        vals.append(0)
    return tuple(vals)


def order_key(indices: tuple[int, int, int], sort_order: list[str]) -> tuple[int, int, int]:
    labels = {"iter": indices[0], "layer": indices[1], "head": indices[2]}
    return tuple(labels[label] for label in sort_order)


def sanitize_tensor(data: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(data).float()
    if tensor.ndim > 2:
        while tensor.ndim > 2:
            tensor = tensor[0]
    return tensor


def compute_block_stats(attn: torch.Tensor, l_f: int, l_h: int, l_w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = l_f * l_h * l_w
    if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
        raise ValueError(f"Attention matrix must be square, got shape {tuple(attn.shape)}")
    if attn.shape[0] != dim:
        raise ValueError(f"Expected edge length {dim}, got {attn.shape[0]}")

    attn = attn.reshape(l_f, l_h * l_w, l_f, l_h * l_w).permute(0, 2, 1, 3).contiguous()
    flat_blocks = attn.view(l_f, l_f, -1)
    means = flat_blocks.mean(dim=-1)
    variances = flat_blocks.var(dim=-1, unbiased=False)
    return means, variances, flat_blocks


def plot_heatmap(matrix: torch.Tensor, out_path: str, title: str, cmap: str, figsize: tuple[int, int]) -> None:
    data = matrix.detach().cpu().numpy()
    plt.figure(figsize=figsize)
    sns.heatmap(data, cmap=cmap, square=True, xticklabels=False, yticklabels=False)
    plt.title(title)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    except Exception as exc:
        print(f"[Error] Failed to save {out_path}: {exc}")
    finally:
        plt.close()


def plot_distribution_grid(
    flat_blocks: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    out_path: str,
    title: str,
    figsize: tuple[float, float],
) -> None:
    l_f = flat_blocks.shape[0]
    fig, axes = plt.subplots(l_f, l_f, figsize=figsize)
    if l_f == 1:
        axes = np.array([[axes]])

    flat_blocks_np = flat_blocks.detach().cpu().numpy()
    means_np = means.detach().cpu().numpy()
    vars_np = variances.detach().cpu().numpy()

    for i in range(l_f):
        for j in range(l_f):
            ax = axes[i, j]
            data = flat_blocks_np[i, j]
            mu = means_np[i, j]
            var = vars_np[i, j]

            ax.hist(data, bins=30, density=True, color="skyblue", edgecolor="none")
            ax.set_xticks([])
            ax.set_yticks([])

            # Annotation
            ax.text(
                0.05,
                0.95,
                f"m:{mu:.2e}\nv:{var:.2e}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6),
            )

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    except Exception as exc:
        print(f"[Error] Failed to save {out_path}: {exc}")
    finally:
        plt.close()


def parse_sort_order(value: str) -> list[str]:
    entries = [entry.strip().lower() for entry in value.split(",") if entry.strip()]
    valid = {"iter", "layer", "head"}
    if not entries:
        return ["iter", "layer", "head"]
    if any(entry not in valid for entry in entries):
        raise argparse.ArgumentTypeError("sort order must contain only iter, layer, head")
    missing = [label for label in ("iter", "layer", "head") if label not in entries]
    return entries + missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Block statistics heatmaps for attention matrices")
    parser.add_argument("--input", type=str, required=True, help="File or directory containing attention scores")
    parser.add_argument("--output", type=str, default=None, help="Output directory (defaults to alongside input files)")
    parser.add_argument("--file-regex", type=str, default=r"score_It(\d+)_L(\d+)_H(\d+)", help="Regex extracting iter/layer/head indices")
    parser.add_argument("--l_f", type=int, required=True, help="Temporal patch count (frames)")
    parser.add_argument("--l_h", type=int, required=True, help="Vertical patch count")
    parser.add_argument("--l_w", type=int, required=True, help="Horizontal patch count")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for heatmaps")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(32.0, 24.0), help="Figure size")
    parser.add_argument("--sort-order", type=parse_sort_order, default="iter,layer,head", help="Comma list priority (default: iter,layer,head)")
    parser.add_argument("--skip-dist", action="store_true", help="Skip generating distribution plots")

    args = parser.parse_args()

    files = collect_files(args.input, SUPPORTED_EXTS, args.file_regex)
    if not files:
        print(f"No supported files found in {args.input}")
        return

    regex = re.compile(args.file_regex)
    sort_order = args.sort_order if isinstance(args.sort_order, list) else parse_sort_order(args.sort_order)
    files.sort(key=lambda path: order_key(parse_indices(path, regex), sort_order))

    out_dir = args.output
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = len(files)
    for idx, path in enumerate(files, 1):
        data = load_score_file(path)
        if data is None:
            continue

        try:
            tensor = sanitize_tensor(data)
            means, variances, flat_blocks = compute_block_stats(tensor, args.l_f, args.l_h, args.l_w)
        except Exception as exc:
            print(f"[Skip] {path}: {exc}")
            continue

        basename = os.path.splitext(os.path.basename(path))[0]
        target_dir = out_dir or os.path.dirname(path)
        mean_path = os.path.join(target_dir, f"{basename}_mean.png")
        var_path = os.path.join(target_dir, f"{basename}_var.png")
        dist_path = os.path.join(target_dir, f"{basename}_dist.png")

        title_prefix = basename.replace("score_", "")
        plot_heatmap(means, mean_path, f"{title_prefix} mean", args.cmap, tuple(args.figsize))
        plot_heatmap(variances, var_path, f"{title_prefix} variance", args.cmap, tuple(args.figsize))

        if not args.skip_dist:
            plot_distribution_grid(flat_blocks, means, variances, dist_path, f"{title_prefix} distribution", tuple(args.figsize))
            print(f"[{idx}/{total}] Saved {mean_path} / {var_path} / {dist_path}")
        else:
            print(f"[{idx}/{total}] Saved {mean_path} / {var_path}")

if __name__ == "__main__":
    main()