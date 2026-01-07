import torch
import sys
import os

# 假设脚本运行在项目根目录或 tests 目录下，确保能导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mask_iter import predict_sparse_attn_mask
from mask_iter_lm import predict_sparse_attn_mask_lm


def main() -> None:
    # 1. 设置随机种子以保证可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("正在初始化测试数据...")

    # 2. 定义测试参数
    # 使用稍大的尺寸以覆盖更多边界情况
    B_hw = 32              # 块大小
    num_block_sqrt = 20    # 块数量 (总大小 640x640)
    num_past_step = 3      # 历史步数
    d_disappear = 5        # 窗口大小
    
    # 时间戳模拟
    timestep_now = 100
    timesteps_past = [98, 95, 90] # 对应 num_past_step = 3
    
    # 权重参数
    alpha = (1.0, 1.0, 0.5) # (w, h, t)
    threshold_inter = 0.5   # 阈值
    
    # 3. 生成随机输入张量 (模拟 0/1 掩码)
    # mask_past: [T, N, B, B]
    mask_past = torch.randint(
        0, 2, 
        (num_past_step, num_block_sqrt, B_hw, B_hw), 
        dtype=torch.uint8
    )
    
    # mask_diag_now: [N, B, B]
    mask_diag_now = torch.randint(
        0, 2, 
        (num_block_sqrt, B_hw, B_hw), 
        dtype=torch.uint8
    )

    # 4. 运行原版函数
    print("正在运行原版 predict_sparse_attn_mask ...")
    start_orig = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_orig = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_orig: start_orig.record()
    
    full_mask_orig, mask_next_orig = predict_sparse_attn_mask(
        B_hw=B_hw,
        num_block_sqrt=num_block_sqrt,
        mask_past=mask_past,
        timesteps_past=timesteps_past,
        timestep_now=timestep_now,
        alpha=alpha,
        threshold_inter=threshold_inter,
        mask_diag_now=mask_diag_now,
        d_disappear=d_disappear,
        num_past_step=num_past_step,
        log_enabled=False # 原版有关闭日志的选项
    )
    
    if end_orig: end_orig.record()

    # 5. 运行低显存优化版函数
    print("正在运行优化版 predict_sparse_attn_mask_lm ...")
    start_lm = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_lm = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_lm: start_lm.record()

    full_mask_lm, mask_next_lm = predict_sparse_attn_mask_lm(
        B_hw=B_hw,
        num_block_sqrt=num_block_sqrt,
        mask_past=mask_past,
        timesteps_past=timesteps_past,
        timestep_now=timestep_now,
        alpha=alpha,
        threshold_inter=threshold_inter,
        mask_diag_now=mask_diag_now,
        d_disappear=d_disappear,
        num_past_step=num_past_step,
        # 注意：优化版没有 log_enabled 参数
    )

    if end_lm: end_lm.record()
    if end_lm: 
        torch.cuda.synchronize()
        print(f"原版耗时: {start_orig.elapsed_time(end_orig):.2f} ms")
        print(f"优化版耗时: {start_lm.elapsed_time(end_lm):.2f} ms")

    # 6. 结果对比
    print("-" * 40)
    print("开始对比结果...")

    # 对比 Full Mask
    diff_full = (full_mask_orig ^ full_mask_lm).sum().item()
    total_full = full_mask_orig.numel()
    is_full_match = (diff_full == 0)
    
    print(f"[Full Mask] 形状: {full_mask_orig.shape}")
    print(f"[Full Mask] 是否完全一致: {is_full_match}")
    if not is_full_match:
        print(f"[Full Mask] 差异像素数: {diff_full} / {total_full} ({diff_full/total_full:.6%})")
    
    # 对比 Next History Mask
    diff_next = (mask_next_orig ^ mask_next_lm).sum().item()
    is_next_match = (diff_next == 0)
    
    print(f"[Next Mask] 形状: {mask_next_orig.shape}")
    print(f"[Next Mask] 是否完全一致: {is_next_match}")
    if not is_next_match:
        print(f"[Next Mask] 差异像素数: {diff_next}")

    print("-" * 40)
    if is_full_match and is_next_match:
        print("✅ 测试通过：两个版本输出完全一致！")
    else:
        print("❌ 测试失败：输出存在差异。")
        # 如果有差异，可能是浮点数计算顺序导致的微小误差在阈值附近发生了翻转
        # 这种情况下，极少量的差异是可以接受的，但理想情况下应为0


if __name__ == "__main__":
    main()