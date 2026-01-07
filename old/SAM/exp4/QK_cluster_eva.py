import argparse
import os
import re
import logging
import numpy as np
import torch
import pandas as pd  # 恢复 pandas 用于结构化数据保存
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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
    
    # 自动合成完整 Q/K (沿着特征维度拼接)
    for prefix in ("Q", "K"):
        if prefix not in components:
            f_key, h_key, w_key = f"{prefix}_f", f"{prefix}_h", f"{prefix}_w"
            if all(k in components for k in (f_key, h_key, w_key)):
                components[prefix] = torch.cat([components[f_key], components[h_key], components[w_key]], dim=-1)
    return components

def _quantize_per_token(array: np.ndarray) -> np.ndarray:
    # array: [L, D], symmetric per-token to int8
    max_abs = np.max(np.abs(array), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    scale = max_abs / 127.0
    q = np.clip(np.rint(array / scale), -127, 127).astype(np.float32) # 转回float方便后续计算
    return q

def evaluate_clustering(data: np.ndarray, sizes: Dict[str, int], cube_params: Tuple[int, int, int, float, int]):
    """
    data: (L, D)
    sizes: {'f': lf, 'h': lh, 'w': lw}
    cube_params: (Cf, Ch, Cw, threshold_quit, Quit_N)
    """
    lf, lh, lw = sizes['f'], sizes['h'], sizes['w']
    cf, ch, cw, threshold, quit_n = cube_params
    D = data.shape[-1]
    
    # 按照 w => h => f 展平规则还原为 (lf, lh, lw, D)
    tensor = data.reshape(lf, lh, lw, D)
    
    l1_pre, cos_pre = [], []
    l1_post, cos_post = [], []
    total_quit = 0
    total_tokens = 0
    
    nf = (lf + cf - 1) // cf
    nh = (lh + ch - 1) // ch
    nw = (lw + cw - 1) // cw
    
    for i in range(nf):
        for j in range(nh):
            for k in range(nw):
                f_s, f_e = i * cf, min((i + 1) * cf, lf)
                h_s, h_e = j * ch, min((j + 1) * ch, lh)
                w_s, w_e = k * cw, min((k + 1) * cw, lw)
                
                cube = tensor[f_s:f_e, h_s:h_e, w_s:w_e, :].reshape(-1, D)
                num_in_cube = cube.shape[0]
                total_tokens += num_in_cube
                
                # 初始计算 (Pre-quit)
                m0 = np.mean(cube, axis=0)
                d0 = np.sum(np.abs(cube - m0), axis=1)
                n0 = np.linalg.norm(cube, axis=1)
                nm0 = np.linalg.norm(m0)
                s0 = np.dot(cube, m0) / (n0 * nm0 + 1e-8) if nm0 > 1e-8 else np.ones(num_in_cube)
                
                l1_pre.extend(d0.tolist())
                cos_pre.extend(s0.tolist())
                
                # Step 2: Iterative Quit Logic
                if quit_n > 0 and np.mean(s0) < threshold:
                    curr_cube = cube.copy()
                    # 循环剔除 Quit_N 个离群点
                    for _ in range(quit_n):
                        if curr_cube.shape[0] <= 1: break
                        # 每次剔除前重新计算当前均值和距离，以找到当前最离群的点
                        curr_m = np.mean(curr_cube, axis=0)
                        curr_d = np.sum(np.abs(curr_cube - curr_m), axis=1)
                        outlier_idx = np.argmax(curr_d)
                        curr_cube = np.delete(curr_cube, outlier_idx, axis=0)
                    
                    total_quit += (num_in_cube - curr_cube.shape[0])
                    
                    # 计算剔除后的指标 (Post-quit)
                    mf = np.mean(curr_cube, axis=0)
                    df = np.sum(np.abs(curr_cube - mf), axis=1)
                    nf_vec = np.linalg.norm(curr_cube, axis=1)
                    nmf = np.linalg.norm(mf)
                    sf = np.dot(curr_cube, mf) / (nf_vec * nmf + 1e-8) if nmf > 1e-8 else np.ones(curr_cube.shape[0])
                    
                    l1_post.extend(df.tolist())
                    cos_post.extend(sf.tolist())
                else:
                    # 不满足剔除条件，Post 指标等于 Pre 指标
                    l1_post.extend(d0.tolist())
                    cos_post.extend(s0.tolist())
                    
    return (np.mean(l1_pre), np.mean(cos_pre), 
            np.mean(l1_post), np.mean(cos_post), 
            total_quit / total_tokens)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Q/K Spatial Locality via Force Clustering")
    parser.add_argument("--input", type=str, required=True, help="Path to .pt files or directory")
    parser.add_argument("--output", type=str, default="cluster_eval.log", help="Path to output log file")
    parser.add_argument("--l_w", type=int, required=True)
    parser.add_argument("--l_h", type=int, required=True)
    parser.add_argument("--l_f", type=int, required=True)
    parser.add_argument("--file_regex", type=str, default=r"(Q|K)_It(\d+)_L(\d+)_H(\d+)")
    parser.add_argument("--quant", action="store_true", help="Whether to use 8-bit quantization")
    parser.add_argument("--save_res", action="store_true", help="Whether to save structured results for plotting")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(args.output), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*50)
    logger.info("Evaluation Parameters (Step 2-v1):")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    logger.info("="*50)

    files = _collect_files(args.input, args.file_regex)
    sizes = {'f': args.l_f, 'h': args.l_h, 'w': args.l_w}
    
    # Step 2-v1 Params: (Cf, Ch, Cw, threshold_quit, Quit_N)
    params = [
        (1, 1, 3, 0.7, 0),  # Param 1: w-direction
        (1, 3, 1, 0.7, 0),  # Param 2: h-direction
        (3, 1, 1, 0.7, 0),  # Param 3: f-direction
        (3, 3, 3, 0.7, 3),  # Param 4: combined with quit
    ]
    
    all_results = []
    logger.info(f"Starting evaluation on {len(files)} files...")
    
    for f_path in files:
        logger.info(f"Processing {os.path.basename(f_path)}...")
        components = _load_components(f_path, torch.device("cpu"))
        
        match = re.search(args.file_regex, os.path.basename(f_path))
        comp_type = match.group(1) if match else "Unknown"
        iter_val = int(match.group(2)) if match else -1
        layer = int(match.group(3)) if match else -1
        head = int(match.group(4)) if match else -1

        for name in ["Q", "K"]:
            if name not in components: continue
            data = components[name].numpy()
            data_eval = _quantize_per_token(data) if args.quant else data
            
            for idx, p in enumerate(params):
                l1_pre, cos_pre, l1_post, cos_post, quit_ratio = evaluate_clustering(data_eval, sizes, p)
                
                res = {
                    "Iter": iter_val,
                    "Layer": layer,
                    "Head": head,
                    "Type": name,
                    "Param_Idx": idx + 1,
                    "Param_Str": f"C({p[0]},{p[1]},{p[2]})_Q{p[4]}",
                    "L1_Pre": l1_pre,
                    "Cos_Pre": cos_pre,
                    "L1_Post": l1_post,
                    "Cos_Post": cos_post,
                    "Quit_Ratio": quit_ratio
                }
                
                # 打印简要日志
                logger.info(f"Iter {iter_val} L{layer} H{head} {name} P{idx+1}: Cos {cos_pre:.4f} -> {cos_post:.4f} (Quit: {quit_ratio:.2%})")
                
                if args.save_res:
                    all_results.append(res)

    if args.save_res and all_results:
        res_path = args.output.replace(".log", "_data.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(res_path, index=False)
        logger.info(f"Structured results saved to {res_path} for plotting.")

    logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()