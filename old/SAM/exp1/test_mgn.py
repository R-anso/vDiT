import torch

from mask_gen_ndiff import generate_diff_sparse_mask
from mask_gen_ndiff_lm import generate_diff_sparse_mask_lm


def main() -> None:
    torch.manual_seed(42)

    batch = 1
    size = 2000  # 2k x 2k
    B_h, B_w = 40, 50  # B_h * B_w 必须等于列数
    alpha = 0.05
    beta = 0.25

    sample = torch.empty(batch, size, size, dtype=torch.float32).uniform_(-10.0, 10.0)

    mask_fp32 = generate_diff_sparse_mask(
        sample,
        B_h=B_h,
        B_w=B_w,
        alpha=alpha,
        beta=beta,
        log_enabled=False,
        mode="ndiff",
    )
    mask_lm = generate_diff_sparse_mask_lm(
        sample,
        B_h=B_h,
        B_w=B_w,
        alpha=alpha,
        beta=beta,
        mode="ndiff",
    )

    diff_count = (mask_fp32 ^ mask_lm).sum().item()
    print(f"掩码是否完全一致：{diff_count == 0}")
    print(f"差异元素数量：{diff_count}")


if __name__ == "__main__":
    main()