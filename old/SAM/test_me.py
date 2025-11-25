import torch
import sys
import os

# 添加 src 目录到路径以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mask_eva import mask_eva
from mask_eva_lm import mask_eva_lm


def test_mask_eva_consistency():
    # 1. 设置随机种子和测试参数
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 128  # 足够大以产生统计意义，又足够小以便快速运行
    
    print(f"正在生成测试数据 (Batch={batch_size}, SeqLen={seq_len})...")

    # 2. 生成随机输入
    # A: 注意力得分 (Logits)
    A = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
    
    # Mask: 随机稀疏掩码 (约 30% 稀疏度)
    mask = (torch.rand(batch_size, seq_len, seq_len) > 0.3).to(torch.bool)

    # 3. 运行原版函数
    print("运行原版 mask_eva...")
    sparsity_orig, retention_orig = mask_eva(
        A, mask, log_enabled=False
    )

    # 4. 运行低显存版函数
    print("运行低显存版 mask_eva_lm...")
    sparsity_lm, retention_lm = mask_eva_lm(A, mask)

    # 5. 比较结果
    print("-" * 40)
    print(f"{'Metric':<15} | {'Original':<12} | {'Low Mem':<12} | {'Diff':<12}")
    print("-" * 40)
    
    diff_sparsity = abs(sparsity_orig - sparsity_lm)
    print(f"{'Sparsity':<15} | {sparsity_orig:<12.6f} | {sparsity_lm:<12.6f} | {diff_sparsity:<12.6e}")
    
    diff_retention = abs(retention_orig - retention_lm)
    print(f"{'Retention':<15} | {retention_orig:<12.6f} | {retention_lm:<12.6f} | {diff_retention:<12.6e}")
    print("-" * 40)

    # 6. 断言检查 (允许极小的浮点误差)
    # 稀疏度应该是完全一致的，因为只是计数
    assert diff_sparsity < 1e-7, "稀疏度计算不一致！"
    
    # 权重保留比例涉及 exp 和 sum，可能会有微小的浮点误差
    assert diff_retention < 1e-6, "权重保留比例计算差异过大！"

    print("✅ 测试通过：两个版本功能一致。")


if __name__ == "__main__":
    test_mask_eva_consistency()