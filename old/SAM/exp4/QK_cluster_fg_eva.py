import argparse
import os
import re
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

_SUPPORTED_EXTS: Tuple[str, ...] = (".pt", ".pth")
_COMPONENT_KEYS: Tuple[str, ...] = ("Q", "K", "Q_w", "Q_h", "Q_f", "K_w", "K_h", "K_f")
_COMPONENT_ALIASES: Dict[str, str] = {
    "q": "Q", "k": "K",
    "q_w": "Q_w", "qw": "Q_w", "q_h": "Q_h", "qh": "Q_h", "q_f": "Q_f", "qf": "Q_f",
    "k_w": "K_w", "kw": "K_w", "k_h": "K_h", "kh": "K_h", "k_f": "K_f", "kf": "K_f",
}

def _collect_files(path: str, regex: Optional[str]) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.lower().endswith(_SUPPORTED_EXTS) else []
    pattern = re.compile(regex) if regex else None
    collected: List[str] = []
    for root, _, files in os.walk(path):
        for fname in files:
            if not fname.lower().endswith(_SUPPORTED_EXTS): continue
            if pattern and not pattern.search(fname): continue
            collected.append(os.path.join(root, fname))
    return sorted(collected)

def _load_components(file_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    raw = torch.load(file_path, map_location=device)
    components: Dict[str, torch.Tensor] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, torch.Tensor): continue
            normalized = _COMPONENT_ALIASES.get(key.lower(), key)
            if normalized in _COMPONENT_KEYS:
                components[normalized] = value.detach().float()
    
    for prefix in ("Q", "K"):
        if prefix not in components:
            f_key, h_key, w_key = f"{prefix}_f", f"{prefix}_h", f"{prefix}_w"
            if all(k in components for k in (f_key, h_key, w_key)):
                components[prefix] = torch.cat([components[f_key], components[h_key], components[w_key]], dim=-1)
    return components

def _quantize_per_token(tensor: torch.Tensor) -> torch.Tensor:
    # tensor: [L, D], symmetric per-token to int8
    max_abs, _ = torch.max(torch.abs(tensor), dim=1, keepdim=True)
    max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(tensor / scale), -127, 127)
    return q

def evaluate_finegrained_ratio(data: torch.Tensor, sizes: Dict[str, int], cube_params: Tuple[int, int, int], threshold: float):
    """
    Vectorized evaluation on GPU/CPU using PyTorch.
    data: (L, D) tensor
    """
    lf, lh, lw = sizes['f'], sizes['h'], sizes['w']
    cf, ch, cw = cube_params
    D = data.shape[-1]
    device = data.device
    
    tensor = data.view(lf, lh, lw, D)
    
    # Calculate padding to make dimensions divisible by cube_params
    pf = (cf - lf % cf) % cf
    ph = (ch - lh % ch) % ch
    pw = (cw - lw % cw) % cw
    
    if pf > 0 or ph > 0 or pw > 0:
        padded = torch.zeros((lf + pf, lh + ph, lw + pw, D), device=device, dtype=data.dtype)
        padded[:lf, :lh, :lw, :] = tensor
        mask = torch.zeros((lf + pf, lh + ph, lw + pw), device=device, dtype=torch.bool)
        mask[:lf, :lh, :lw] = True
    else:
        padded = tensor
        mask = torch.ones((lf, lh, lw), device=device, dtype=torch.bool)

    nf, nh, nw = padded.shape[0] // cf, padded.shape[1] // ch, padded.shape[2] // cw
    
    # Reshape into cubes: (nf, nh, nw, cf*ch*cw, D)
    cubes = padded.view(nf, cf, nh, ch, nw, cw, D)
    cubes = cubes.permute(0, 2, 4, 1, 3, 5, 6).reshape(nf, nh, nw, cf * ch * cw, D)
    
    cube_mask = mask.view(nf, cf, nh, ch, nw, cw)
    cube_mask = cube_mask.permute(0, 2, 4, 1, 3, 5).reshape(nf, nh, nw, cf * ch * cw)
    
    # Compute mean per cube, ignoring masked elements
    cube_sums = (cubes * cube_mask.unsqueeze(-1)).sum(dim=3)
    cube_counts = cube_mask.sum(dim=3, keepdim=True).clamp(min=1)
    cube_means = cube_sums / cube_counts
    
    # Cosine similarity: (A . B) / (|A| * |B|)
    dot_product = (cubes * cube_means.unsqueeze(3)).sum(dim=-1)
    norm_a = torch.norm(cubes, dim=-1)
    norm_b = torch.norm(cube_means, dim=-1)
    
    cos_sim = dot_product / (norm_a * norm_b.unsqueeze(-1) + 1e-8)
    # If mean vector is zero, similarity is 1 (as per original logic)
    cos_sim = torch.where(norm_b.unsqueeze(-1) < 1e-8, torch.ones_like(cos_sim), cos_sim)
    
    # Mean cosine similarity per cube (only valid tokens)
    cube_cos_sums = (cos_sim * cube_mask).sum(dim=3)
    cube_cos_means = cube_cos_sums / cube_counts.squeeze(-1)
    
    good_cubes = (cube_cos_means >= threshold).sum().item()
    total_cubes = nf * nh * nw
    
    return good_cubes / total_cubes

def plot_finegrained_results(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    groups = df.groupby(['Layer', 'Head', 'Type'])
    
    all_params = sorted(df['Cube_Param'].unique())
    all_thresholds = sorted(df['Threshold'].unique())
    
    # Mapping: Params -> Markers, Thresholds -> Colors
    marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    marker_map = {p: marker_list[i % len(marker_list)] for i, p in enumerate(all_params)}
    
    color_list = plt.cm.tab10.colors
    color_map = {t: color_list[i % len(color_list)] for i, t in enumerate(all_thresholds)}
    
    for (layer, head, qk_type), group in groups:
        group = group.sort_values('Iter')
        plt.figure(figsize=(14, 9))
        
        for p_str in all_params:
            for thresh in all_thresholds:
                sub = group[(group['Cube_Param'] == p_str) & (group['Threshold'] == thresh)]
                if not sub.empty:
                    plt.plot(sub['Iter'], sub['Ratio'], 
                             label=f'{p_str} T={thresh:.1f}', 
                             marker=marker_map[p_str], 
                             color=color_map[thresh],
                             alpha=0.6, markersize=5)
            
        plt.xlabel('Iteration (Timestep Index)')
        plt.ylabel('Ratio of Suitable Cubes')
        plt.title(f'Layer {layer} Head {head} {qk_type} Fine-grained Clustering Ratio')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
        plt.ylim(-0.05, 1.05)
        
        filename = f"Layer_{layer}_Head_{head}_{qk_type}_QK_finegrained_cluster_ratio.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

def plot_summary_results(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    def get_n(p_str):
        nums = re.findall(r'\d+', p_str)
        n = 1
        for x in nums: n *= int(x)
        return n

    # 1. 计算所有 Iter 的均值
    summary = df.groupby(['Layer', 'Head', 'Type', 'Cube_Param', 'Threshold'])['Ratio'].mean().reset_index()
    
    # 2. 将 Q 和 K 对齐到同一行
    summary_pivot = summary.pivot(index=['Layer', 'Head', 'Cube_Param', 'Threshold'], columns='Type', values='Ratio').reset_index()
    if 'Q' not in summary_pivot.columns: summary_pivot['Q'] = 0.0
    if 'K' not in summary_pivot.columns: summary_pivot['K'] = 0.0
    
    # 3. 计算加速比
    summary_pivot['N'] = summary_pivot['Cube_Param'].apply(get_n)
    summary_pivot['f_Q'] = 1 - summary_pivot['Q'] * (1 - 1/summary_pivot['N'])
    summary_pivot['f_K'] = 1 - summary_pivot['K'] * (1 - 1/summary_pivot['N'])
    
    summary_pivot['Speedup_Q'] = 1 / (summary_pivot['f_Q'] + 1e-8)
    summary_pivot['Speedup_K'] = 1 / (summary_pivot['f_K'] + 1e-8)
    summary_pivot['Speedup_A'] = 1 / (summary_pivot['f_Q'] * summary_pivot['f_K'] + 1e-8)
    
    # Define styles
    cube_params = sorted(summary_pivot['Cube_Param'].unique())
    param_colors = plt.cm.Set3.colors
    param_color_map = {p: param_colors[i % len(param_colors)] for i, p in enumerate(cube_params)}
    
    hatch_map = {
        'Q': '///', 'K': '\\\\\\',
        'Speedup_Q': '///', 'Speedup_K': '\\\\\\', 'Speedup_A': 'xxx'
    }

    groups = summary_pivot.groupby(['Layer', 'Head'])
    for (layer, head), group in groups:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 1. Ratio 子图
        ratio_df = group.melt(id_vars=['Threshold', 'Cube_Param'], value_vars=['Q', 'K'], var_name='Comp', value_name='Ratio')
        ratio_df = ratio_df.sort_values(['Threshold', 'Cube_Param', 'Comp'])
        ratio_df['Label'] = ratio_df['Comp'] + "_" + ratio_df['Cube_Param']
        
        pivoted_ratio = ratio_df.pivot(index='Threshold', columns='Label', values='Ratio')
        pivoted_ratio.plot(kind='bar', ax=ax1, edgecolor='black', linewidth=0.5)
        
        ax1.set_ylabel('Mean Ratio')
        ax1.set_title(f'Layer {layer} Head {head} - Clustering Ratios (Q & K)')
        ax1.set_ylim(0, 1.2)
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        
        for i, container in enumerate(ax1.containers):
            label = pivoted_ratio.columns[i]
            comp, cp = label.rsplit('_', 1) # Use rsplit to handle labels correctly
            for bar in container:
                bar.set_facecolor(param_color_map.get(cp, 'gray'))
                bar.set_hatch(hatch_map.get(comp, ''))
            ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=7, rotation=90)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
        
        # 2. Speedup 子图
        speed_df = group.melt(id_vars=['Threshold', 'Cube_Param'], value_vars=['Speedup_Q', 'Speedup_K', 'Speedup_A'], var_name='Type', value_name='Speedup')
        speed_df = speed_df.sort_values(['Threshold', 'Cube_Param', 'Type'])
        speed_df['Label'] = speed_df['Type'] + "_" + speed_df['Cube_Param']
        
        pivoted_speed = speed_df.pivot(index='Threshold', columns='Label', values='Speedup')
        pivoted_speed.plot(kind='bar', ax=ax2, edgecolor='black', linewidth=0.5)
        
        ax2.set_yscale('log')
        ax2.set_ylabel('Theoretical Speedup (x, Log Scale)')
        ax2.set_title(f'Layer {layer} Head {head} - Theoretical Speedups (Q, K, A)')
        ax2.grid(axis='y', which='both', linestyle=':', alpha=0.6)
        
        max_speedup = 0
        for i, container in enumerate(ax2.containers):
            label = pivoted_speed.columns[i]
            stype, cp = label.rsplit('_', 1) # Use rsplit to correctly extract Cube_Param
            for bar in container:
                bar.set_facecolor(param_color_map.get(cp, 'gray'))
                bar.set_hatch(hatch_map.get(stype, ''))
                h = bar.get_height()
                if not np.isnan(h): max_speedup = max(max_speedup, h)
            ax2.bar_label(container, fmt='%.1f', padding=3, fontsize=7, rotation=90)
        
        ax2.set_ylim(1, max_speedup * 2.0) # Log scale cannot start at 0, use 1 as base
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
        
        plt.tight_layout()
        filename = f"Layer_{layer}_Head_{head}_QK_summary_combined.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-grained Q/K Clustering Ratio")
    parser.add_argument("--input", type=str, required=True, help="Path to .pt files or directory")
    parser.add_argument("--log_dir", type=str, default="fg_logs", help="Directory for log and csv files")
    parser.add_argument("--plot_dir", type=str, default="fg_plots", help="Directory for plots")
    parser.add_argument("--l_w", type=int, required=True)
    parser.add_argument("--l_h", type=int, required=True)
    parser.add_argument("--l_f", type=int, required=True)
    parser.add_argument("--file_regex", type=str, default=r"(Q|K)_It(\d+)_L(\d+)_H(\d+)")
    parser.add_argument("--quant", action="store_true", help="Whether to use 8-bit quantization")
    parser.add_argument("--plot", action="store_true", help="Whether to generate plots")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "fg_cluster_eval.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    files = _collect_files(args.input, args.file_regex)
    sizes = {'f': args.l_f, 'h': args.l_h, 'w': args.l_w}
    cube_params_list = [(3, 3, 3), (2, 3, 3), (3, 2, 2), (2, 2, 2)]
    thresholds = [0.7, 0.8, 0.9]
    
    all_results = []
    logger.info(f"Starting fine-grained evaluation on {len(files)} files...")
    
    for f_path in files:
        components = _load_components(f_path, device)
        match = re.search(args.file_regex, os.path.basename(f_path))
        iter_val = int(match.group(2)) if match else -1
        layer = int(match.group(3)) if match else -1
        head = int(match.group(4)) if match else -1

        for name in ["Q", "K"]:
            if name not in components: continue
            data = components[name]
            data_eval = _quantize_per_token(data) if args.quant else data
            
            for cp in cube_params_list:
                cp_str = f"C{cp}"
                for thresh in thresholds:
                    ratio = evaluate_finegrained_ratio(data_eval, sizes, cp, thresh)
                    all_results.append({
                        "Iter": iter_val,
                        "Layer": layer,
                        "Head": head,
                        "Type": name,
                        "Cube_Param": cp_str,
                        "Threshold": thresh,
                        "Ratio": ratio
                    })
                    logger.info(f"Iter {iter_val} L{layer} H{head} {name} {cp_str} Thresh {thresh}: Ratio {ratio:.2%}")

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.log_dir, "fg_cluster_data.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        if args.plot:
            logger.info("Generating plots...")
            plot_finegrained_results(df, args.plot_dir)
            plot_summary_results(df, args.plot_dir)
            logger.info(f"Plots saved to {args.plot_dir}")

if __name__ == "__main__":
    main()
