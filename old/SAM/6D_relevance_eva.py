import os
import json
import argparse
import numpy as np
import torch
import re
import sys

def get_6d_shape(L: int, B_h: int, B_w: int) -> tuple[int, int, int, int, int, int]:
    """
    根据总长度 L 和空间维度 B_h, B_w 计算 6D 形状。
    L = l_f * l_h * l_w
    """
    if L % (B_h * B_w) != 0:
        raise ValueError(f"L={L} cannot be divided by B_h*B_w ({B_h}*{B_w}={B_h*B_w})")
    
    l_f = L // (B_h * B_w)
    l_h = B_h
    l_w = B_w
    
    # 映射关系: (Q_f, Q_h, Q_w, K_f, K_h, K_w)
    return (l_f, l_h, l_w, l_f, l_h, l_w)

def calc_q_relevance(tensor_6d: torch.Tensor, dim_idx: int, use_cv: bool = True) -> float:
    """
    计算 Q 方向维度的相关性。
    Args:
        use_cv: 是否使用变异系数 (Coefficient of Variation)。
    """
    # 1. 计算标准差
    stds = torch.std(tensor_6d, dim=dim_idx)
    
    # # debug: 查看原始张量以及标准差输出信息
    # # 除dim_idx外其他维度均取0，查看单个切片
    # debug_slice = [0]*6
    # debug_slice[dim_idx] = slice(None)  # 在 dim_idx 维度上取所有元素
    # sample_tensor = tensor_6d[tuple(debug_slice)]
    # sample_mean = torch.mean(sample_tensor).item()
    # sample_std = torch.std(sample_tensor).item()
    # sample_cv = sample_std / (sample_mean + 1e-9)

    # print(f"\n[DEBUG Q-Dim {dim_idx}] Sample slice [0,0,...,:,...,0,0]:")
    # print(f"  Tensor: {sample_tensor.tolist()}")
    # print(f"  Mean: {sample_mean:.6f}")
    # print(f"  Std: {sample_std:.6f}")
    # print(f"  CV: {sample_cv:.6f}")

    if use_cv:
        # 2. 计算均值
        means = torch.mean(tensor_6d, dim=dim_idx)
        # 3. 计算变异系数 CV = std / mean
        # 添加 epsilon 防止除以零 (对于 mask 掉的区域 mean 可能为 0)
        cvs = stds / (means + 1e-9)
        return float(cvs.mean().item())
    else:
        return float(stds.mean().item())

def calc_k_relevance(tensor_6d: torch.Tensor, dim_idx: int, use_cv: bool = True) -> tuple[float, float, float]:
    """
    计算 K 方向维度的相关性（基于均值）。
    返回:
        (k_q_style_metric, k_ratio_mean_std, k_pos_std)
        - k_q_style_metric: 与Q相同方法（CV或Std）
        - k_ratio_mean_std: 各元素/均值 的比值在样本维上的标准差的均值
        - k_pos_std: 与均值最接近元素的索引在样本间的标准差
    """
    # 把待分析维度移到最后，其余展平
    dims = list(range(6))
    dims.remove(dim_idx)
    dims.append(dim_idx)
    X = tensor_6d.permute(*dims).reshape(-1, tensor_6d.shape[dim_idx])  # [M, N]

    # 与Q相同的方法
    stds_k = torch.std(X, dim=-1)  # [M]
    if use_cv:
        means_k_scalar = torch.mean(X, dim=-1)  # [M]
        k_q_style_metric = float((stds_k / (means_k_scalar + 1e-9)).mean().item())
    else:
        k_q_style_metric = float(stds_k.mean().item())

    # 与均值最接近的元素索引
    means_k = X.mean(dim=-1, keepdim=True)  # [M,1]
    mean_diff = torch.abs(X - means_k)      # [M,N]
    nearest_mean_idx = torch.argmin(mean_diff, dim=-1).float()  # [M]
    k_pos_std = float(nearest_mean_idx.std().item())

    # 比值改为对均值的比值
    ratios = X / (means_k + 1e-9)           # [M,N]
    k_ratio_mean_std = float(ratios.std(dim=0).mean().item())

    return k_q_style_metric, k_ratio_mean_std, k_pos_std

def evaluate_6d_relevance(
    score: torch.Tensor, 
    B_h: int, 
    B_w: int,
    is_softmaxed: bool = True,
    use_cv: bool = True
) -> list[float]:
    """
    执行 6D 相关性评估。
    Returns:
        [Qf, Qh, Qw, Kf_q, Kf_ratio, Kf_pos, Kh_q, Kh_ratio, Kh_pos, Kw_q, Kw_ratio, Kw_pos]
        共 12 个值
    """
    # 1. 预处理张量形状
    if score.ndim == 4:
        print("Warning: Input is 4D, using score[0, 0] for evaluation.")
        mat = score[0, 0]
    elif score.ndim == 3:
        print("Warning: Input is 3D, using score[0] for evaluation.")
        mat = score[0]
    elif score.ndim == 2:
        mat = score
    else:
        raise ValueError(f"Unsupported score shape: {score.shape}")
    
    L = mat.shape[0]
    if mat.shape[1] != L:
        raise ValueError(f"Score matrix must be square, got {mat.shape}")

    # 2. 数据准备 (Softmax)
    if not is_softmaxed:
        mat = torch.softmax(mat.to(dtype=torch.float32), dim=-1)
    else:
        mat = mat.to(dtype=torch.float32)

    # 3. 映射到 6D
    shape_6d = get_6d_shape(L, B_h, B_w)
    tensor_6d = mat.view(shape_6d)
    
    results_q = []
    for dim in range(3):
        r = calc_q_relevance(tensor_6d, dim, use_cv=use_cv)
        results_q.append(r)
        
    # K 维度结果分组：先K(Q)的f/h/w，再K(R)的f/h/w，再K(P)的f/h/w
    kq_list, kr_list, kp_list = [], [], []
    for dim in range(3, 6):
        kq, kr, kp = calc_k_relevance(tensor_6d, dim, use_cv=use_cv)
        kq_list.append(kq)
        kr_list.append(kr)
        kp_list.append(kp)

    return results_q + kq_list + kr_list + kp_list

def _load_score_file(path: str, device: str = "cpu") -> torch.Tensor | None:
    """加载分数文件"""
    try:
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".json"):
            with open(path, "r") as f:
                d = json.load(f)
                data = np.array(d["scores"]) if "scores" in d else np.array(d)
        elif path.endswith(".txt"):
            data = np.loadtxt(path)
        else:
            return None
        return torch.from_numpy(data).to(device)
    except Exception as e:
        print(f"[Error] Could not load {path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6-Dimension Relevance Evaluation for Wan2.2 Attention")
    parser.add_argument("--score_dir", type=str, default="./attn_analysis/attn_score/F53/cond", help="Directory containing score files")
    parser.add_argument("--B_h", type=int, default=22, help="Block height (l_h)")
    parser.add_argument("--B_w", type=int, default=40, help="Block width (l_w)")
    parser.add_argument("--type", type=str, choices=["A", "S"], default="S", help="Input type: 'A' (Raw Score) or 'S' (Softmaxed)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--log_file", type=str, default="6d_relevance.log", help="Output log file path")
    parser.add_argument("--use_cv", action="store_true", help="Use Coefficient of Variation for relevance calculation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.score_dir):
        print(f"Error: Directory {args.score_dir} does not exist.")
        sys.exit(1)
        
    fname_pattern = re.compile(r"score_It(\d+)_L(\d+)_H(\d+)")
    
    files = [f for f in os.listdir(args.score_dir) if f.endswith((".npy", ".json", ".txt")) and "score" in f]
    files.sort()
    
    print(f"Found {len(files)} files in {args.score_dir}")
    print(f"Processing on {args.device}, Input Type: {args.type}, Use CV: {args.use_cv}")
    
    # 更新日志头
    with open(args.log_file, "w", encoding="utf-8") as f:
        header = (
            f"{'Iter':<5} {'Layer':<5} {'Head':<5} | "
            f"{'Q_f':<8} {'Q_h':<8} {'Q_w':<8} | "
            f"{'K(Q)_f':<8} {'K(Q)_h':<8} {'K(Q)_w':<8} | "
            f"{'K(R)_f':<8} {'K(R)_h':<8} {'K(R)_w':<8} | "
            f"{'K(P)_f':<8} {'K(P)_h':<8} {'K(P)_w':<8}\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")
        print(header.strip())

    for fname in files:
        match = fname_pattern.search(fname)
        if not match:
            continue
            
        iter_idx, layer_idx, head_idx = match.groups()
        file_path = os.path.join(args.score_dir, fname)
        
        score_tensor = _load_score_file(file_path, args.device)
        if score_tensor is None:
            continue
            
        try:
            is_softmaxed = (args.type == "S")
            
            coeffs = evaluate_6d_relevance(
                score_tensor, 
                B_h=args.B_h, 
                B_w=args.B_w, 
                is_softmaxed=is_softmaxed,
                use_cv=args.use_cv
            )
            
            # coeffs: [Qf, Qh, Qw, Kf_q, Kf_r, Kf_p, Kh_q, Kh_r, Kh_p, Kw_q, Kw_r, Kw_p]
            log_line = (
                f"{iter_idx:<5} {layer_idx:<5} {head_idx:<5} | "
                f"{coeffs[0]:<8.4f} {coeffs[1]:<8.4f} {coeffs[2]:<8.4f} | "
                f"{coeffs[3]:<8.4f} {coeffs[4]:<8.4f} {coeffs[5]:<8.4f} | "
                f"{coeffs[6]:<8.4f} {coeffs[7]:<8.4f} {coeffs[8]:<8.4f} | "
                f"{coeffs[9]:<8.2f} {coeffs[10]:<8.2f} {coeffs[11]:<8.2f}"
            )
            
            print(log_line)
            
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
                
        except Exception as e:
            err_msg = f"Error processing {fname}: {e}"
            print(err_msg)
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(f"{iter_idx:<5} {layer_idx:<5} {head_idx:<5} | ERROR: {e}\n")

    print(f"Done. Results saved to {args.log_file}")