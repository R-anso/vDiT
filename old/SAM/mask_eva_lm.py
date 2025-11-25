import os
import time
import torch


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
    
    step_info = f"--- Iteration Step: {iter_step.item()} ---\n" if iter_step is not None else ""
    
    log_message = (
        f"{step_info}"
        f"[{timestamp}] mask_eva_lm call:\n"
        f"  - Batch Size: {batch_size}, Sequence Length: {seq_len}\n"
        f"  - Sparsity: {sparsity:.2%}\n"
        f"  - Weight Retention: {weight_retention:.2%}\n\n"
    )
    
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(log_message)


def mask_eva_lm(
    A: torch.Tensor,
    mask: torch.Tensor,
    *,
    log_enabled: bool = False,
    log_dir: str = "./log",
    iter_step: torch.Tensor = None,
) -> tuple[float, float]:
    """
    低显存版：评估稀疏掩码的性能（带日志）。
    
    优化点：
    1. 避免实例化完整的 Softmax 矩阵 [B, L, L]。
    2. 使用 LogSumExp 计算归一化因子。
    3. 仅计算被 Mask 保留元素的指数值。
    """
    # 1. 输入验证
    if A.ndim != 3 or mask.ndim != 3 or A.shape != mask.shape:
        raise ValueError(
            f"输入张量 A 和 mask 必须是相同形状的3维张量, "
            f"得到 A: {A.shape}, mask: {mask.shape}"
        )
    
    batch_size, seq_len, _ = A.shape
    
    # 2. 计算稀疏度 (Sparsity)
    # 使用 float32 累加避免溢出，但不需要转换整个 mask
    # sum() 返回的是标量或小张量，显存开销极小
    total_elements = batch_size * seq_len * seq_len
    non_zero_count = mask.sum().item()
    avg_sparsity = 1.0 - (non_zero_count / total_elements)

    # 3. 计算权重保留比例 (Weight Retention)
    # 目标: sum(Softmax(A) * mask)
    # Softmax(A)_ij = exp(A_ij) / sum_k(exp(A_ik))
    # Retention_i = sum_j(exp(A_ij) * mask_ij) / sum_k(exp(A_ik))
    
    # 步骤 3.1: 计算每行的 LogSumExp (归一化因子的对数)
    # shape: [Batch, L, 1]
    # 为了数值稳定性，通常 A - max(A) 再 exp，但 torch.logsumexp 已经处理了
    log_norm_factor = torch.logsumexp(A, dim=-1, keepdim=True)
    
    # 步骤 3.2: 仅计算保留部分的权重
    # 我们利用 mask 筛选出需要的 A_ij
    # 为了避免显存爆炸，我们不计算全量 exp(A)，而是：
    # Retention = sum(exp(A_masked - log_norm))
    
    # 使用 masked_select 提取元素，展平为 1D 向量
    # 这比保留 [B, L, L] 的零填充矩阵要省显存得多（当稀疏度高时）
    A_masked = torch.masked_select(A, mask.to(torch.bool))
    
    # 对应的归一化因子也需要展开对齐
    # log_norm_factor: [B, L, 1] -> expand to [B, L, L] -> masked_select
    log_norm_expanded = log_norm_factor.expand(-1, -1, seq_len)
    log_norm_masked = torch.masked_select(log_norm_expanded, mask.to(torch.bool))
    
    # 计算保留的概率值
    # P_masked = exp(A_masked - log_norm_masked)
    # 使用原地操作减法
    # 注意：这里会创建一个与非零元素数量相同的临时张量
    retention_values = torch.exp(A_masked - log_norm_masked)
    
    # 求和得到总保留权重，再除以总行数 (Batch * L)
    # 因为每行的 Softmax 和为 1，所以所有行的总权重理论上是 (Batch * L)
    # 平均保留比例 = 总保留权重 / (Batch * L)
    total_retained_weight = retention_values.sum().item()
    avg_weight_retention = total_retained_weight / (batch_size * seq_len)

    # 4. 记录日志 (保留原版功能)
    if log_enabled:
        _log_evaluation_summary(
            log_dir=log_dir,
            sparsity=avg_sparsity,
            weight_retention=avg_weight_retention,
            batch_size=batch_size,
            seq_len=seq_len,
            iter_step=iter_step,
        )

    return avg_sparsity, avg_weight_retention


if __name__ == "__main__":
    # 简单测试
    torch.manual_seed(42)
    batch, L = 2, 8
    A_sample = torch.randn(batch, L, L)
    mask_sample = (torch.rand(batch, L, L) > 0.5).to(torch.bool)
    
    s_val, r_val = mask_eva_lm(
        A_sample, 
        mask_sample, 
        log_enabled=True, 
        log_dir="./test_logs",
        iter_step=torch.tensor(100)
    )
    print(f"Sparsity: {s_val:.2%}")
    print(f"Retention: {r_val:.2%}")