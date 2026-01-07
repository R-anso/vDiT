import os
import time
import torch
import torch.nn.functional as F
import numpy as np


def _ensure_log_dir(log_dir: str) -> None:
    """确保日志目录存在。"""
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def _log_evaluation_summary(
    log_dir: str,
    sparsity: float,
    weight_retention: float,
    batch_size: int,
    seq_len: int,
    iter_step: torch.Tensor = None,
) -> None:
    """将评估结果写入日志文件。"""
    _ensure_log_dir(log_dir)
    summary_path = os.path.join(log_dir, "mask_evaluation.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    log_message = (
        f"--- Iteration Step: {iter_step.item()} ---\n"
        f"[{timestamp}] mask_eva call:\n"
        f"  - Batch Size: {batch_size}, Sequence Length: {seq_len}\n"
        f"  - Sparsity: {sparsity:.2%}\n"
        f"  - Weight Retention: {weight_retention:.2%}\n\n"
    )
    
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(log_message)


def mask_eva(
    A: torch.Tensor,
    mask: torch.Tensor,
    *,
    log_enabled: bool = False,
    log_dir: str = "./log",
    iter_step: torch.Tensor = None,
) -> tuple[float, float]:
    """
    评估稀疏掩码的性能。

    Args:
        A (torch.Tensor): 注意力得分矩阵 (Softmax之前)，形状为 [Batch, L_token, L_token]。
        mask (torch.Tensor): 布尔或0/1稀疏掩码，形状与A相同。
        log_enabled (bool): 是否启用日志记录。
        log_dir (str): 日志文件目录。

    Returns:
        tuple[float, float]: 一个包含两个标量值的元组：
            - 平均稀疏度 (float)
            - 平均权重保留比例 (float)
    """
    # 1. 输入验证
    if A.ndim != 3 or mask.ndim != 3 or A.shape != mask.shape:
        raise ValueError(
            f"输入张量 A 和 mask 必须是相同形状的3维张量, "
            f"得到 A: {A.shape}, mask: {mask.shape}"
        )
    
    batch_size, seq_len, _ = A.shape
    mask = mask.to(dtype=torch.float32, device=A.device)

    # 2. 计算稀疏度
    # 计算每行中非零元素的比例（即密度）
    density_per_row = mask.sum(dim=-1) / seq_len
    # 稀疏度 = 1 - 密度
    sparsity_per_row = 1.0 - density_per_row
    # 计算整个批次的平均稀疏度
    avg_sparsity = sparsity_per_row.mean().item()

    # 3. 计算权重保留比例
    # 对A矩阵进行Softmax操作得到注意力权重S
    S = F.softmax(A, dim=-1)
    
    # 将掩码应用于S
    D = S * mask
    
    # 计算每一行被保留的权重之和
    retained_weights_per_row = D.sum(dim=-1)
    
    # 计算整个批次的平均权重保留比例
    avg_weight_retention = retained_weights_per_row.mean().item()

    # 4. 记录日志
    if log_enabled:
        _log_evaluation_summary(
            log_dir=log_dir,
            sparsity=avg_sparsity,
            weight_retention=avg_weight_retention,
            batch_size=batch_size,
            seq_len=seq_len,
            iter_step=iter_step,
        )

    # 5. 返回结果
    return avg_sparsity, avg_weight_retention


if __name__ == "__main__":
    # 创建一个示例
    torch.manual_seed(42)
    batch, L = 2, 8
    
    # 随机生成注意力得分矩阵 A
    A_sample = torch.randn(batch, L, L)
    
    # 随机生成一个稀疏掩码 M (约50%的稀疏度)
    mask_sample = (torch.rand(batch, L, L) > 0.5).to(torch.bool)
    
    print("="*30)
    print("运行掩码评估函数...")
    print("="*30)
    
    sparsity_val, retention_val = mask_eva(
        A_sample, 
        mask_sample, 
        log_enabled=True, 
        log_dir="./log"
    )
    
    print(f"输入A形状: {A_sample.shape}")
    print(f"输入mask形状: {mask_sample.shape}")
    print("-" * 30)
    print(f"计算得到的平均稀疏度: {sparsity_val:.2%}")
    print(f"计算得到的平均权重保留比例: {retention_val:.2%}")
    print("\n日志已写入 './test_logs/mask_evaluation.log'")

    # 验证日志内容
    with open("./test_logs/mask_evaluation.log", "r", encoding="utf-8") as f:
        print("\n日志文件内容预览:")
        print("-" * 20)
        print(f.read())