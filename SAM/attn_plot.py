import os
import json
import argparse
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def load_score_file(path: str) -> np.ndarray | None:
    """
    加载分数文件，支持 npy, json, txt, pt, pth。
    返回 numpy 数组。
    """
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return None

    try:
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                # 兼容直接列表或字典格式
                data = np.array(d["scores"]) if isinstance(d, dict) and "scores" in d else np.array(d)
        elif path.endswith(".txt"):
            data = np.loadtxt(path)
        elif path.endswith((".pt", ".pth")):
            data = torch.load(path, map_location="cpu")
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            else:
                data = np.asarray(data)
        else:
            print(f"[Error] Unsupported file format: {path}")
            return None
        
        return data
    except Exception as e:
        print(f"[Error] Could not load {path}: {e}")
        return None

def plot_heatmap(
    data: np.ndarray | torch.Tensor,
    out_path: str,
    title: str = "Attention Heatmap",
    cmap: str = "viridis",
    figsize: tuple = (10, 8)
):
    """
    绘制热力图并保存。
    """
    # 1. 数据预处理
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # 降维处理：如果是 3D/4D 张量，这里简单处理为取第一个样本或第一头
    if data.ndim > 2:
        # print(f"[Info] Input shape {data.shape} is > 2D. Taking the first slice for visualization.")
        while data.ndim > 2:
            data = data[0]
            
    # 2. 绘图
    plt.figure(figsize=figsize)
    
    # 使用 seaborn 绘制热力图
    sns.heatmap(data, cmap=cmap, square=True, xticklabels=False, yticklabels=False)
    
    plt.title(title)
    
    # 3. 保存
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    try:
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        # print(f"[Saved] {out_path}")
    except Exception as e:
        print(f"[Error] Failed to save heatmap to {out_path}: {e}")
    finally:
        plt.close()

def collect_files(path: str, extensions: tuple, regex: str | None = None) -> list[str]:
    """收集文件，支持目录递归和正则过滤"""
    files = []
    if os.path.isfile(path):
        if path.lower().endswith(extensions):
             files.append(path)
    elif os.path.isdir(path):
        pattern = re.compile(regex) if regex else None
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    if pattern and not pattern.search(filename):
                        continue
                    files.append(os.path.join(root, filename))
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Plot Attention Heatmap from Score Files")
    
    # 输入输出设置
    parser.add_argument("--input", type=str, default="./attn_analysis/attn_score/F53_pic/cond", help="Path to the input score file OR directory containing score files")
    parser.add_argument("--output", type=str, default="./attn_analysis/attn_heatmap/F53_pic/cond", help="Path to output directory (if input is dir) or file (if input is file). Defaults to input location.")
    parser.add_argument("--file_regex", type=str, default=r"score_It(\d+)_L(\d+)_H(\d+)", help="Regex pattern to filter files in directory (e.g., 'score_It0_.*')")
    
    # 处理选项
    parser.add_argument("--softmax", action="store_true", help="Apply softmax to the input matrix before plotting")
    parser.add_argument("--log", action="store_true", help="Apply log1p to the input matrix before plotting")
    parser.add_argument("--dim", type=int, default=-1, help="Dimension to apply softmax along (default: -1)")
    
    # 块显示选项
    parser.add_argument("--fbo", action="store_true", help="Only plot the first data block (size l_h*l_w x l_h*l_w)")
    parser.add_argument("--swap_hw", action="store_true", help="Swap H and W dimensions (transpose spatial grid) within the block")
    parser.add_argument("--l_h", type=int, default=22, help="Height of the spatial latent (used for block size calculation)")
    parser.add_argument("--l_w", type=int, default=40, help="Width of the spatial latent (used for block size calculation)")

    # 绘图选项
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for the heatmap")
    parser.add_argument("--title", type=str, default=None, help="Title of the plot (defaults to filename)")

    args = parser.parse_args()

    supported_exts = (".npy", ".json", ".txt", ".pt", ".pth")
    
    # 1. 收集文件
    print(f"Scanning {args.input}...")
    files = collect_files(args.input, supported_exts, args.file_regex)
    
    if not files:
        print(f"No supported files found in {args.input}")
        return

    print(f"Found {len(files)} files. Processing...")
    
    is_dir_input = os.path.isdir(args.input)
    count = 0

    for fpath in files:
        try:
            # 2. 加载数据
            data = load_score_file(fpath)
            if data is None:
                continue

            # 3. 数据变换
            tensor_data = torch.from_numpy(data).float()

            if args.softmax:
                tensor_data = torch.softmax(tensor_data, dim=args.dim)
            
            if args.log:
                tensor_data = torch.log1p(tensor_data)

            # 截取第一个数据块
            if args.fbo:
                block_size = args.l_h * args.l_w
                # 检查维度是否足够
                if tensor_data.shape[-2] < block_size or tensor_data.shape[-1] < block_size:
                    print(f"[Warning] Data shape {tensor_data.shape} is smaller than block size {block_size}x{block_size}. Skipping slicing.")
                else:
                    # 对最后两个维度进行切片，兼容 (N, N) 或 (B, H, N, N) 等情况
                    tensor_data = tensor_data[..., :block_size, :block_size]

                    if args.swap_hw:
                        # 假设当前是 (..., H, W, H, W) 也就是 H 是慢轴，W 是快轴
                        # 我们想要变成 (..., W, H, W, H) 也就是 W 是慢轴，H 是快轴
                        try:
                            # 1. Reshape: 将平铺的维度还原为 (H, W)
                            # 保存前面的维度
                            prefix_shape = tensor_data.shape[:-2]
                            # 展开最后两维: (..., H, W, H, W)
                            view_shape = prefix_shape + (args.l_h, args.l_w, args.l_h, args.l_w)
                            tensor_data = tensor_data.reshape(view_shape)

                            # 2. Permute: 交换 H 和 W
                            # 维度索引：... H_q(-4), W_q(-3), H_k(-2), W_k(-1)
                            # 目标：... W_q, H_q, W_k, H_k
                            ndim = tensor_data.ndim
                            perm_indices = list(range(ndim - 4)) + [ndim - 3, ndim - 4, ndim - 1, ndim - 2]
                            tensor_data = tensor_data.permute(perm_indices)

                            # 3. Reshape back: 重新平铺为 (..., W*H, W*H)
                            tensor_data = tensor_data.reshape(prefix_shape + (block_size, block_size))
                        except Exception as e:
                            print(f"[Error] Failed to swap H/W: {e}. Check if l_h={args.l_h}, l_w={args.l_w} match the block size.")

            # 4. 确定输出路径
            fname = os.path.basename(fpath)
            base_name = os.path.splitext(fname)[0]
            
            if args.output:
                if is_dir_input:
                    # 输入是目录，args.output 被视为输出目录
                    # 保持相对目录结构（可选），这里简单地全部放到输出目录
                    out_path = os.path.join(args.output, base_name + ".png")
                else:
                    # 输入是单文件
                    if os.path.isdir(args.output) or args.output.endswith(os.sep):
                         out_path = os.path.join(args.output, base_name + ".png")
                    else:
                         out_path = args.output
            else:
                # 默认：保存在原文件同级目录
                out_path = os.path.join(os.path.dirname(fpath), base_name + ".png")
            
            plot_title = args.title if args.title else f"{base_name}"

            # 5. 绘制
            plot_heatmap(
                data=tensor_data,
                out_path=out_path,
                title=plot_title,
                cmap=args.cmap
            )
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{len(files)} files...", end="\r")
                
        except Exception as e:
            print(f"\n[Fail] Processing {fpath}: {e}")

    print(f"\nDone. Successfully processed {count} files.")

if __name__ == "__main__":
    main()