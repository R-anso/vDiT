import math
import torch
import os
import json
import numpy as np
import traceback
import re

def mask_gen(
    scores: torch.Tensor,
    mode: str = "threshold",
    param: float | dict = 0.5,
) -> torch.Tensor:
    """
    统一掩码生成函数，集成 threshold, topk, block_topk, ndiff 四种方法。

    Args:
        scores: 输入分数张量，通常为[B, H, L_q, L_k] 或 [B, L_q, L_k] 或 [L_q, L_k]
        mode: 生成模式 ("threshold", "topk", "block_topk", "ndiff")。
        param: 核心参数，类型可变。
               - threshold: float (阈值) 或 dict {'threshold': float}
               - topk: float (保留比例) 或 dict {'topk': float}
               - block_topk: dict {'topk': float, 'block_parts': int}
               - ndiff: dict {'alpha': float, 'beta': float, 'B_h': int, 'B_w': int}

    Returns:
        torch.Tensor: 生成的布尔掩码 (bool dtype)。
    """
    mode = mode.lower()

    # 1. NDIFF Mode (基于局部差异的稀疏掩码)
    if mode == "ndiff":
        if not isinstance(param, dict):
             raise ValueError("Mode 'ndiff' requires param to be a dictionary containing alpha, beta, B_h, and B_w.")
        
        p_alpha = param.get("alpha")
        p_beta = param.get("beta")
        p_Bh = param.get("B_h")
        p_Bw = param.get("B_w")
        
        if p_alpha is None or p_beta is None or p_Bh is None or p_Bw is None:
            raise ValueError("Mode 'ndiff' requires alpha, beta, B_h, and B_w parameters in param dict.")

        # 形状适配: ndiff 核心逻辑基于 [Batch, Rows, Cols]
        original_shape = scores.shape
        if scores.ndim == 4:
            b, h, q, k = scores.shape
            scores_in = scores.reshape(b * h, q, k)
        elif scores.ndim == 3:
            scores_in = scores
        elif scores.ndim == 2:
            # [Rows, Cols] -> [1, Rows, Cols]
            scores_in = scores.unsqueeze(0)
        else:
            raise ValueError(f"Mode 'ndiff' expects 2D, 3D or 4D tensor, got {scores.ndim}D")

        # --- 分块处理逻辑 ---
        # 将大矩阵切分为 B_h * B_w 的小块，独立执行 ndiff
        N, H, W = scores_in.shape
        B_h, B_w = int(p_Bh), int(p_Bw)
        
        # 1. Padding: 确保长宽能被 B_h, B_w 整除
        pad_h = (B_h - H % B_h) % B_h
        pad_w = (B_w - W % B_w) % B_w
        
        if pad_h > 0 or pad_w > 0:
            # 使用 0.0 填充，假设 scores 为概率或分数，避免影响 TopK 选择
            scores_padded = torch.nn.functional.pad(scores_in, (0, pad_w, 0, pad_h), value=0.0)
        else:
            scores_padded = scores_in
            
        H_p, W_p = scores_padded.shape[-2:]
        n_h = H_p // B_h
        n_w = W_p // B_w
        
        # 2. Reshape to Blocks
        # [N, H_p, W_p] -> [N, n_h, B_h, n_w, B_w] -> [N, n_h, n_w, B_h, B_w]
        # -> [N, n_h * n_w, B_h * B_w]
        # 这样每个 B_h * B_w 的块就变成了 core 函数的一个 "row" (即一个独立的样本)
        blocks = scores_padded.view(N, n_h, B_h, n_w, B_w).permute(0, 1, 3, 2, 4).reshape(N, n_h * n_w, B_h * B_w)
        
        # 3. Core Execution
        # core 函数会对 blocks 的每一行 (即每个小块) 计算统计量并生成掩码
        mask_blocks = _generate_ndiff_core(blocks, B_h, B_w, float(p_alpha), float(p_beta))
        
        # 4. Reshape Back & Crop
        # [N, n_h * n_w, B_h * B_w] -> [N, n_h, n_w, B_h, B_w] -> [N, n_h, B_h, n_w, B_w] -> [N, H_p, W_p]
        mask_padded = mask_blocks.view(N, n_h, n_w, B_h, B_w).permute(0, 1, 3, 2, 4).reshape(N, H_p, W_p)
        
        if pad_h > 0 or pad_w > 0:
            mask = mask_padded[:, :H, :W]
        else:
            mask = mask_padded
        
        # 5. Restore Original Shape
        if scores.ndim == 4:
            mask = mask.reshape(original_shape)
        elif scores.ndim == 2:
            mask = mask.squeeze(0)
        return mask

    # 2. Threshold Mode (阈值截断)
    if mode == "threshold":
        val = param
        if isinstance(param, dict):
            val = param.get("threshold", 0.5)
        return scores > float(val)

    # 3. TopK Mode (全局 TopK)
    if mode == "topk":
        ratio = param
        if isinstance(param, dict):
            ratio = param.get("topk", 0.5)
        
        ratio = float(ratio)
        k = max(1, int(scores.size(-1) * ratio))
        
        # 直接在原 dtype 上操作
        topk_indices = torch.topk(scores, k, dim=-1, largest=True, sorted=True).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        return mask

    # 4. Block TopK Mode (分块 TopK)
    if mode == "block_topk":
        ratio = 0.5
        parts = 1
        
        if isinstance(param, dict):
            ratio = param.get("topk", ratio)
            parts = param.get("block_parts", parts)
            
            # 尝试从 B_h/B_w 推断 parts (如果未提供 block_parts 但提供了 B_h/B_w)
            if "block_parts" not in param and "B_h" in param and "B_w" in param:
                # 假设是正方形分块，或者只看行/列的比例
                # 这里简单假设针对 L_q 和 L_k 进行等分
                # 注意：block_topk 的 parts 通常指将长宽各切几刀
                # 如果 B_h 是块的高度，那么 parts = L_q / B_h
                l_q = scores.size(-2)
                b_h = param["B_h"]
                if b_h > 0:
                    parts = max(1, int(math.ceil(l_q / b_h)))
        elif isinstance(param, (float, int)):
            ratio = float(param)
            
        return _generate_block_topk(scores, float(ratio), int(parts))

    raise ValueError(f"Unknown mode: {mode}")


def _generate_ndiff_core(tensor: torch.Tensor, B_h: int, B_w: int, alpha: float, beta: float) -> torch.Tensor:
    """
    ndiff 核心逻辑。
    """
    batch_size, num_rows, num_cols = tensor.shape
    
    if B_h <= 0 or B_w <= 0:
        raise ValueError("B_h 与 B_w 需为正整数")
    if B_h * B_w != num_cols:
        raise ValueError(f"B_h * B_w ({B_h}*{B_w}={B_h*B_w}) must equal num_cols ({num_cols})")
    if alpha <= 0:
        raise ValueError("alpha 必须为正数")
    if not (0.0 <= beta <= 0.5):
        raise ValueError("beta 需位于 [0, 0.5]")

    # 使用 reshape 避免拷贝
    segments = tensor.reshape(batch_size, num_rows, B_h, B_w)

    # 1. 计算统计量 (Max/Diff)
    # 保持原数据类型
    segment_max = segments.max(dim=-1).values
    row_max = segment_max.max(dim=-1, keepdim=True).values
    diff = row_max - segment_max

    # 2. 计算保留数量 (Grades)
    # 优化: 利用覆盖特性，避免创建 (diff > alpha) & (diff <= 2*alpha) 这样的临时布尔张量
    grades = torch.zeros_like(diff, dtype=torch.int8)
    # 先处理最宽的范围 (diff <= 3*alpha)，填 1
    grades.masked_fill_(diff <= 3 * alpha, 1)
    # 再处理中间范围 (diff <= 2*alpha)，填 2 (覆盖掉满足条件的 1)
    grades.masked_fill_(diff <= 2 * alpha, 2)
    # 最后处理最窄范围 (diff <= alpha)，填 3 (覆盖掉满足条件的 2)
    grades.masked_fill_(diff <= alpha, 3)

    # 计算 keep_counts，中间计算使用原数据类型以保持一致性
    keep_counts = torch.ceil(grades.to(tensor.dtype) * (beta * B_w)).to(torch.int32)
    keep_counts.clamp_(max=B_w)

    max_keep = int(keep_counts.max().item())
    if max_keep == 0:
        return torch.zeros((batch_size, num_rows, num_cols), device=tensor.device, dtype=torch.bool)

    # 3. 执行 TopK (在原 dtype 上进行)
    topk_indices = torch.topk(segments, k=max_keep, dim=-1, largest=True, sorted=True).indices
    
    # 4. 生成 Mask
    rank_range = torch.arange(max_keep, device=tensor.device, dtype=keep_counts.dtype)
    # keep_counts: [B, Rows, B_h] -> [B, Rows, B_h, 1]
    # rank_range: [1, 1, 1, max_keep]
    valid_mask = rank_range.view(1, 1, 1, max_keep) < keep_counts.unsqueeze(-1)

    result = torch.zeros((batch_size, num_rows, num_cols), device=tensor.device, dtype=torch.bool)
    segment_view = result.reshape(batch_size, num_rows, B_h, B_w)
    
    # scatter_ 支持广播，将 valid_mask 填入对应位置
    segment_view.scatter_(-1, topk_indices, valid_mask)
    
    return result


def _generate_block_topk(scores: torch.Tensor, param: float, block_parts: int) -> torch.Tensor:
    """
    分块 TopK 核心逻辑。
    """
    n = max(1, block_parts)
    mask_sel = torch.zeros_like(scores, dtype=torch.bool)
    
    len_q, len_k = scores.size(-2), scores.size(-1)
    q_step = math.ceil(len_q / n)
    k_step = math.ceil(len_k / n)

    for qi in range(n):
        qs = qi * q_step
        qe = min(qs + q_step, len_q)
        if qs >= len_q: break
        
        for ki in range(n):
            ks = ki * k_step
            ke = min(ks + k_step, len_k)
            if ks >= len_k: break
            
            # 切片
            block = scores[..., qs:qe, ks:ke]
            if block.numel() == 0: continue
            
            block_top_k = max(1, int(block.size(-1) * param))
            
            # 在 block 内部进行 topk
            topk_idx = torch.topk(block, block_top_k, dim=-1).indices
            
            # 优化: 直接在 mask_sel 的切片视图上进行 scatter_
            # 避免了分配临时的 block_mask 张量，也避免了后续的拷贝操作
            mask_sel[..., qs:qe, ks:ke].scatter_(-1, topk_idx, True)
            
    return mask_sel


def save_mask_to_file(
    mask: torch.Tensor | np.ndarray,
    out_path: str,
    out_fmt: str = "npy"
) -> None:
    """
    将掩码保存到文件，支持 npy, txt, json 格式。
    参考 save_mask.py 实现，包含原子写入操作。
    """
    try:
        # 1. 统一转换为 numpy bool 数组
        if isinstance(mask, torch.Tensor):
            mask_bool = mask.detach().cpu().numpy().astype(np.bool_)
        else:
            mask_bool = np.asarray(mask).astype(np.bool_)

        # 2. 准备路径
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        fmt = out_fmt.lower()
        if not out_path.lower().endswith(f".{fmt}"):
            out_path += f".{fmt}"
            
        tmp_path = out_path + f".tmp"

        # 3. 根据格式写入临时文件
        if fmt == "npy":
            # 修改: 使用文件对象写入，防止 np.save 自动追加 .npy 后缀导致文件名不匹配
            with open(tmp_path, "wb") as f:
                np.save(f, mask_bool)
        elif fmt == "txt":
            with open(tmp_path, "w", encoding="utf-8") as f:
                # 如果是多维数组，展平或按行写入，这里简单处理为按行写入 0/1
                if mask_bool.ndim <= 1:
                    f.write(" ".join("1" if v else "0" for v in mask_bool) + "\n")
                else:
                    # 展平除第一维以外的维度，保持每行一个样本/头/行
                    flattened = mask_bool.reshape(mask_bool.shape[0], -1)
                    for row in flattened:
                        f.write(" ".join("1" if v else "0" for v in row) + "\n")
        elif fmt == "json":
            obj = {
                "shape": tuple(mask_bool.shape),
                "dtype": "bool",
                # 转为 uint8 (0/1) 以压缩 JSON 体积并提高兼容性
                "mask": mask_bool.astype(np.uint8).tolist(),
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=None) # 不缩进以减小体积
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        # 4. 原子替换
        if os.path.exists(out_path):
            os.remove(out_path)
        os.replace(tmp_path, out_path)
        print(f"[Saved] {out_path}")

    except Exception as e:
        print(f"[Error] Failed to save mask to {out_path}: {e}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_score_file(path: str) -> torch.Tensor | None:
    """辅助函数：加载分数文件"""
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
        return torch.from_numpy(data)
    except Exception as e:
        print(f"[Warn] Could not load {path}: {e}")
        return None


if __name__ == "__main__":

    # 1. 输入设置
    # 假设结构为: ./attn_score/{FOLDER_NAME}/{KEY}/score_It{iter}_L{layer}_H{head}.{fmt}
    FOLDER_NAME = "default"
    KEY = "cond"             # 'cond' or 'uncond'
    
    # 输入根目录 (读取分数) 和 输出根目录 (保存掩码)
    SCORE_ROOT = "./attn_score"
    MASK_ROOT = "./attn_mask"

    INPUT_FMTS = (".npy", ".json", ".txt")
    FNAME_BASE = "score" # 模板基
    
    # 文件名解析正则 (参考 attn_tools.py: score_It{iter}_L{layer}_H{head})
    FNAME_PATTERN = re.compile(r"score_It(\d+)_L(\d+)_H(\d+)")
    
    # 2. 掩码生成参数
    MODE = "ndiff" 
    
    # Case A: Threshold / TopK
    # PARAM = 0.5 
    
    # Case B: Block TopK
    # PARAM = 0.3
    # block_parts 现在通过 PARAM 字典传入，或者由 B_h/B_w 推断
    
    # Case C: NDIFF
    PARAM = {
        "alpha": 0.1,   # 差异阈值
        "beta": 0.1,    # 保留比例
        "B_h": 22,      # 块高度
        "B_w": 40       # 块宽度
    }
    
    # 4. 输出设置
    OUT_FORMAT = "npy"  # 'npy', 'txt', 'json'

    # 构建输入路径: ./attn_score/a/key
    in_dir = os.path.join(SCORE_ROOT, FOLDER_NAME, KEY)
    
    # 构建输出路径: ./attn_mask/a/key
    out_dir = os.path.join(MASK_ROOT, FOLDER_NAME, KEY)
    
    if not os.path.exists(in_dir):
        print(f"Directory not found: {in_dir}")
        exit(1)

    print(f"Scanning {in_dir}...")
    print(f"Mode: {MODE}, Param: {PARAM}")

    files = [f for f in os.listdir(in_dir) if f.endswith(INPUT_FMTS) and f.startswith(FNAME_BASE)]
    
    if not files:
        print("No score files found.")
    
    os.makedirs(out_dir, exist_ok=True)
    
    count = 0
    for fname in files:
        # 解析文件名信息
        match = FNAME_PATTERN.search(fname)
        if not match:
            print(f"[Skip] Cannot parse filename: {fname}")
            continue
            
        iter_idx, layer_idx, head_idx = match.groups()
        file_path = os.path.join(in_dir, fname)
        
        # 1. 加载分数
        scores = _load_score_file(file_path)
        if scores is None:
            continue
            
        # 2. 生成掩码
        try:
            mask = mask_gen(
                scores=scores,
                mode=MODE,
                param=PARAM
            )
            
            # 3. 保存掩码
            out_fname = f"mask_It{iter_idx}_L{layer_idx}_H{head_idx}_M{MODE}"
            
            save_path = os.path.join(out_dir, out_fname)
            save_mask_to_file(mask, save_path, OUT_FORMAT)
            count += 1
            
        except Exception as e:
            print(f"[Fail] Processing {fname}: {e}")
            traceback.print_exc()

    print(f"Done. Processed {count} files.")