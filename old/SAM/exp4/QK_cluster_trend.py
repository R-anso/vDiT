import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_qk_trends(csv_path, output_dir):
    # 读取数据
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 按照 Layer, Head, Type (Q/K) 进行分组
    # 这样每个 Head 的 Q 和 K 会分别画一张图
    groups = df.groupby(['Layer', 'Head', 'Type'])
    
    print(f"Starting to plot {len(groups)} figures...")
    
    for (layer, head, qk_type), group in groups:
        # 确保 Iter 是有序的，防止折线乱跳
        group = group.sort_values('Iter')
        
        plt.figure(figsize=(10, 6))
        
        # 提取 4 个参数的数据
        p1 = group[group['Param_Idx'] == 1]
        p2 = group[group['Param_Idx'] == 2]
        p3 = group[group['Param_Idx'] == 3]
        p4 = group[group['Param_Idx'] == 4]
        
        # 绘制 5 条折线
        # 1. W-direction (Param 1)
        plt.plot(p1['Iter'], p1['Cos_Pre'], label='W-dir (1,1,3)', marker='o', alpha=0.7)
        
        # 2. H-direction (Param 2)
        plt.plot(p2['Iter'], p2['Cos_Pre'], label='H-dir (1,3,1)', marker='s', alpha=0.7)
        
        # 3. F-direction (Param 3)
        plt.plot(p3['Iter'], p3['Cos_Pre'], label='F-dir (3,1,1)', marker='^', alpha=0.7)
        
        # 4. Combined Pre (Param 4 剔除前)
        plt.plot(p4['Iter'], p4['Cos_Pre'], label='Combined Pre (3,3,3)', 
                 linestyle='--', color='gray', alpha=0.6)
        
        # 5. Combined Post (Param 4 剔除后) - 重点突出
        plt.plot(p4['Iter'], p4['Cos_Post'], label='Combined Post (3,3,3, Q3)', 
                 linewidth=2.5, color='red', marker='D', markersize=4)
        
        # 图表修饰
        plt.xlabel('Iteration (Timestep Index)')
        plt.ylabel('Mean Cosine Similarity')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize='small')
        
        # 限制 Y 轴范围，方便观察差异（通常相似度在 0-1 之间）
        plt.ylim(min(0.3, group['Cos_Pre'].min() * 0.9), 1.02)
        
        # 保存图片
        filename = f"Layer_{layer}_Head_{head}_{qk_type}_QK_cluster_trend.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QK Clustering Trends from CSV data")
    parser.add_argument("--csv", type=str, required=True, help="Path to the _data.csv file")
    parser.add_argument("--out_dir", type=str, default="cluster_plots", help="Directory to save PNGs")
    args = parser.parse_args()
    
    plot_qk_trends(args.csv, args.out_dir)