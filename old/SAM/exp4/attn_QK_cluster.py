import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class Attn_Cluster_Cfg:
    """Configuration for Attention Clustering (Step 4)"""
    def __init__(
        self,
        enable: bool = False,
        q_cf: int = 3, q_ch: int = 3, q_cw: int = 3, q_threshold: float = 0.85,
        k_cf: int = 3, k_ch: int = 3, k_cw: int = 3, k_threshold: float = 0.85,
        
        check_point_interval: int = 5,
        out_dir: str = "./attn_analysis/cluster_results",
        
        # [New] Quantization flag
        quantize: bool = False,
        
        # skip一些敏感区间
        skip_start_iters: int = 5,
        skip_end_iters: int = 5,
        total_steps: int = 50,
    ):
        self.enable = enable
        self.q_cf, self.q_ch, self.q_cw = q_cf, q_ch, q_cw
        self.q_threshold = q_threshold
        self.k_cf, self.k_ch, self.k_cw = k_cf, k_ch, k_cw
        self.k_threshold = k_threshold
        self.check_point_interval = check_point_interval
        self.out_dir = out_dir
        self.skip_start_iters = skip_start_iters
        self.skip_end_iters = skip_end_iters
        self.total_steps = total_steps
        self.quantize = quantize

class Attn_Cluster_Manager:
    """Manages clustering masks and records results for Step 4"""
    def __init__(self, cfg: Attn_Cluster_Cfg, num_layers: int, num_heads: int):
        self.cfg = cfg
        self.num_layers = num_layers
        self.num_heads = num_heads
        # masks[(layer_idx, head_idx)] = {'Q': mask, 'K': mask}
        self.masks: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        
        if self.cfg.enable:
            q_p = f"Q{cfg.q_cf}{cfg.q_ch}{cfg.q_cw}_T{cfg.q_threshold}"
            k_p = f"K{cfg.k_cf}{cfg.k_ch}{cfg.k_cw}_T{cfg.k_threshold}"
            param_dir = f"{q_p}_{k_p}_CI{cfg.check_point_interval}"
            self.full_out_dir = os.path.join(cfg.out_dir, param_dir)
            os.makedirs(self.full_out_dir, exist_ok=True)

    def _quantize_per_token(self, data: torch.Tensor) -> torch.Tensor:
        """
        对 Tensor 进行 Token 级别的量化与反量化 (8-bit)
        data: (L, D)
        """
        # 找到每个 token 的绝对值最大值: (L, 1)
        max_val = torch.max(torch.abs(data), dim=-1, keepdim=True)[0]
        # 计算量化系数: 将 max 映射到 127
        scale = 127.0 / (max_val + 1e-8)
        
        # 量化到 -127 ~ 127 并取整
        q_data = torch.round(data * scale).clamp(-127, 127)
        
        # 反量化回浮点数
        return q_data / scale

    def is_active_iter(self, iter_idx: int) -> bool:
        """判断当前迭代步是否应该开启聚类算法"""
        if not self.cfg.enable or iter_idx is None:
            return False
        
        # 判定是否在跳过区间内
        if iter_idx < self.cfg.skip_start_iters:
            return False
        if iter_idx >= (self.cfg.total_steps - self.cfg.skip_end_iters):
            return False
            
        return True

    def should_update(self, iter_idx: int) -> bool:
        if not self.is_active_iter(iter_idx):
            return False
        # 在活跃区间内，按间隔更新掩码
        return (iter_idx % self.cfg.check_point_interval == 0)

    def get_masks(self, layer_idx: int, head_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        return self.masks.get((layer_idx, head_idx))

    def update_mask(self, layer_idx: int, head_idx: int, q: torch.Tensor, k: torch.Tensor, sizes: Dict[str, int]):
        if not self.cfg.enable:
            return

        q_mask = self._evaluate_good_cubes(q, sizes, self.cfg.q_threshold, self.cfg.q_cf, self.cfg.q_ch, self.cfg.q_cw)
        k_mask = self._evaluate_good_cubes(k, sizes, self.cfg.k_threshold, self.cfg.k_cf, self.cfg.k_ch, self.cfg.k_cw)
        
        self.masks[(layer_idx, head_idx)] = {'Q': q_mask, 'K': k_mask}

    def _evaluate_good_cubes(self, data: torch.Tensor, sizes: Dict[str, int], threshold: float, cf: int, ch: int, cw: int) -> torch.Tensor:
        # 如果开启量化，在此处抹去幅度差异
        if self.cfg.quantize:
            max_val = torch.max(torch.abs(data), dim=-1, keepdim=True)[0]
            scale = 127.0 / (max_val + 1e-8)
            # data 现在是幅度归一化到 127 附近的“纯方向”向量
            data = torch.round(data * scale).clamp(-127, 127)
            
        lf, lh, lw = sizes['f'], sizes['h'], sizes['w']
        D = data.shape[-1]
        device = data.device
        
        tensor = data.view(lf, lh, lw, D)
        pf, ph, pw = (cf - lf % cf) % cf, (ch - lh % ch) % ch, (cw - lw % cw) % cw
        
        if pf > 0 or ph > 0 or pw > 0:
            padded = torch.zeros((lf + pf, lh + ph, lw + pw, D), device=device, dtype=data.dtype)
            padded[:lf, :lh, :lw, :] = tensor
            mask = torch.zeros((lf + pf, lh + ph, lw + pw), device=device, dtype=torch.bool)
            mask[:lf, :lh, :lw] = True
        else:
            padded, mask = tensor, torch.ones((lf, lh, lw), device=device, dtype=torch.bool)

        nf, nh, nw = padded.shape[0] // cf, padded.shape[1] // ch, padded.shape[2] // cw
        cubes = padded.view(nf, cf, nh, ch, nw, cw, D).permute(0, 2, 4, 1, 3, 5, 6).reshape(nf, nh, nw, cf * ch * cw, D)
        cube_mask = mask.view(nf, cf, nh, ch, nw, cw).permute(0, 2, 4, 1, 3, 5).reshape(nf, nh, nw, cf * ch * cw)
        
        cube_sums = (cubes * cube_mask.unsqueeze(-1)).sum(dim=3)
        cube_counts = cube_mask.sum(dim=3, keepdim=True).clamp(min=1)
        cube_means = cube_sums / cube_counts
        
        dot_product = (cubes * cube_means.unsqueeze(3)).sum(dim=-1)
        cos_sim = dot_product / (torch.norm(cubes, dim=-1) * torch.norm(cube_means, dim=-1).unsqueeze(-1) + 1e-8)
        cube_cos_means = (cos_sim * cube_mask).sum(dim=3) / cube_counts.squeeze(-1)
        
        return cube_cos_means >= threshold

    def apply_clustering(self, data: torch.Tensor, sizes: Dict[str, int], cube_mask_to_apply: torch.Tensor, cf: int, ch: int, cw: int) -> torch.Tensor:
        # 记录原始 scale 用于后续恢复幅度
        if self.cfg.quantize:
            max_val = torch.max(torch.abs(data), dim=-1, keepdim=True)[0]
            orig_scales = 127.0 / (max_val + 1e-8)
            # 进入 INT8 域，消除幅度差异
            data_int8 = torch.round(data * orig_scales).clamp(-127, 127)
        else:
            data_int8 = data # 浮点模式下直接使用
            
        lf, lh, lw = sizes['f'], sizes['h'], sizes['w']
        D = data.shape[-1]
        device = data.device
        
        # 将 data_int8 变形为立方体
        tensor = data_int8.view(lf, lh, lw, D)
        pf, ph, pw = (cf - lf % cf) % cf, (ch - lh % ch) % ch, (cw - lw % cw) % cw
        
        if pf > 0 or ph > 0 or pw > 0:
            padded = torch.zeros((lf + pf, lh + ph, lw + pw, D), device=device, dtype=data.dtype)
            padded[:lf, :lh, :lw, :] = tensor
            mask = torch.zeros((lf + pf, lh + ph, lw + pw), device=device, dtype=torch.bool)
            mask[:lf, :lh, :lw] = True
        else:
            padded, mask = tensor, torch.ones((lf, lh, lw), device=device, dtype=torch.bool)

        nf, nh, nw = padded.shape[0] // cf, padded.shape[1] // ch, padded.shape[2] // cw
        # Reshape into cubes
        cubes = padded.view(nf, cf, nh, ch, nw, cw, D).permute(0, 2, 4, 1, 3, 5, 6).reshape(nf, nh, nw, cf * ch * cw, D)
        cube_mask_tokens = mask.view(nf, cf, nh, ch, nw, cw).permute(0, 2, 4, 1, 3, 5).reshape(nf, nh, nw, cf * ch * cw)
        
        # 计算均值向量（基于抹去幅度差异后的向量）
        cube_means = (cubes * cube_mask_tokens.unsqueeze(-1)).sum(dim=3) / cube_mask_tokens.sum(dim=3, keepdim=True).clamp(min=1)
        
        # 聚类替换：将 cube 内部所有 token 统一为该 cube 的均值方向
        clustered_cubes = torch.where(cube_mask_to_apply.unsqueeze(-1).unsqueeze(-1), cube_means.unsqueeze(3), cubes)
        
        # 还原形状
        restored_int8 = clustered_cubes.view(nf, nh, nw, cf, ch, cw, D).permute(0, 3, 1, 4, 2, 5, 6).reshape(nf*cf, nh*ch, nw*cw, D)
        restored_int8 = restored_int8[:lf, :lh, :lw, :].reshape(lf * lh * lw, D)
        
        # [关键步骤]：根据每个 Token 原始的 scale 反量化回去。
        # 如果该 Cube 聚类了，那么这 27 个 Token 会有完全相同的方向 (cube_means)，
        # 但通过乘以各自的 (1/orig_scale)，它们保留了原始的 token-wise 幅度差异。
        if self.cfg.quantize:
            return restored_int8 / orig_scales
        else:
            return restored_int8

    def record_metrics(self, layer_idx: int, head_idx: int, iter_idx: Optional[int], q_ratio: float, k_ratio: float):
        if (layer_idx, head_idx) not in self.results:
            self.results[(layer_idx, head_idx)] = []
            
        nq = self.cfg.q_cf * self.cfg.q_ch * self.cfg.q_cw
        nk = self.cfg.k_cf * self.cfg.k_ch * self.cfg.k_cw
        f_q = 1 - q_ratio * (1 - 1/nq)
        f_k = 1 - k_ratio * (1 - 1/nk)
        speedup = 1 / (f_q * f_k + 1e-8)
        
        self.results[(layer_idx, head_idx)].append({
            'Iter': iter_idx if iter_idx is not None else -1,
            'Q_Ratio': q_ratio, 'K_Ratio': k_ratio, 'Speedup': speedup
        })

    def save_and_plot(self):
        """Save all results to CSV and generate plots for each head"""
        if not self.results:
            return
            
        all_data = []
        for (l, h), res_list in self.results.items():
            for res in res_list:
                row = {'Layer': l, 'Head': h}
                row.update(res)
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        csv_path = os.path.join(self.full_out_dir, "clustering_metrics.csv")
        df.to_csv(csv_path, index=False)
        
        # Plotting
        for (l, h), group in df.groupby(['Layer', 'Head']):
            group = group.sort_values('Iter')
            # 稍微增加宽度以容纳外部图例
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # 绘制 Ratio (左轴)
            lns1 = ax1.plot(group['Iter'], group['Q_Ratio'], label='Q Clustering Ratio', 
                           color='tab:blue', marker='o', linestyle='-', alpha=0.8)
            lns2 = ax1.plot(group['Iter'], group['K_Ratio'], label='K Clustering Ratio', 
                           color='tab:green', marker='s', linestyle='--', alpha=0.8)
            
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Clustering Ratio', color='black')
            ax1.set_ylim(-0.05, 1.05)
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            # 绘制 Speedup (右轴)
            ax2 = ax1.twinx()
            lns3 = ax2.plot(group['Iter'], group['Speedup'], label='Theoretical Speedup (A)', 
                           color='tab:red', linewidth=2, marker='D')
            ax2.set_ylabel('Speedup (x)', color='tab:red')
            ax2.set_yscale('log')
            # 添加 1x 基准线
            ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
            
            # 合并图例并移到绘图区右侧，避免遮挡曲线
            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.1, 1.0), borderaxespad=0.)
            
            # 增加标题信息
            title_str = (f"Layer {l} Head {h} Clustering Trend\n"
                         f"Q: {self.cfg.q_cf}x{self.cfg.q_ch}x{self.cfg.q_cw}, T={self.cfg.q_threshold} | "
                         f"K: {self.cfg.k_cf}x{self.cfg.k_ch}x{self.cfg.k_cw}, T={self.cfg.k_threshold}")
            plt.title(title_str, fontsize=10, pad=15)
            
            # 调整布局
            plt.tight_layout()
            
            plot_path = os.path.join(self.full_out_dir, f"Layer_{l}_Head_{h}_QK_cluster_acceleration_ratio.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
