import torch

# 全局开关
SPARSE_ATTN_ENABLED = False
LOW_MEMORY_MODE = True # This mode will disable logging
EVALUATE_MASK = True
NDIFF_SPARSE_MODE = 'ndiff'

# 块内掩码生成配置 (mask_gen_ndiff.py)
B_H = 22
B_W = 40
# 较大的值会保留更多细节。
ALPHA = 1
BETA = 0.2

# 块间掩码预测配置 (mask_iter.py)
NUM_PAST_STEP = 2

def distance_function(dist_tensor: torch.Tensor) -> torch.Tensor:
    """
    定义应用于距离的惩罚函数。
    该函数将被 mask_iter 用于计算历史掩码的权重。
    默认使用二次惩罚，距离越远，权重衰减越快。
    
    您可以修改此函数以实现不同的衰减策略，例如：
    - 线性衰减: return dist_tensor
    - 指数衰减: return torch.exp(dist_tensor) - 1.0
    """
    return torch.pow(dist_tensor, 2)

# 预测非对角块时，用于加权历史信息的权重因子 (alpha_w, alpha_h, alpha_t)。
ALPHA_ITER = (0.2, 0.2, 1)

# 用于非对角块的阈值。
THRESHOLD_INTER = 0.25

# 非对角块与对角线的最大保留距离。
D_DISAPPEAR = 5

# 日志配置
LOG_ENABLED = False
LOG_DIR = f"./log/mask_eva/A{ALPHA}_B{BETA}_Ni{NUM_PAST_STEP}_T{THRESHOLD_INTER}_D{D_DISAPPEAR}_M{NDIFF_SPARSE_MODE}"

ATTN_SAVE_ENABLED = False
ATTN_SAVE_FORMAT = "npy"  # 'npy' | 'txt' | 'json'
ATTN_SAVE_DIR = "./attn_analysis/attn_mask/{NDIFF_SPARSE_MODE}/A{ALPHA}_B{BETA}_Ni{NUM_PAST_STEP}_T{THRESHOLD_INTER}_D{D_DISAPPEAR}"

# 将配置打包成字典
def get_sparse_attn_config():
    """
    将所有配置打包成一个字典，方便传递给模型。
    """
    if not SPARSE_ATTN_ENABLED:
        return {'enabled': False}

    return {
        'enabled': True,
        'low_memory_mode': LOW_MEMORY_MODE,
        'evaluate_mask': EVALUATE_MASK,
        'ndiff_mode': NDIFF_SPARSE_MODE,
        'log_enabled': LOG_ENABLED,
        'log_dir': LOG_DIR,
        'attn_save_enabled': ATTN_SAVE_ENABLED,
        'attn_save_dir': ATTN_SAVE_DIR,
        'attn_save_format': ATTN_SAVE_FORMAT,

        # 块内配置
        'B_h': B_H,
        'B_w': B_W,
        'B_hw': B_H * B_W, # 自动计算块边长
        'alpha': ALPHA,
        'beta': BETA,

        # 块间配置
        'num_past_step': NUM_PAST_STEP,
        'dist_func': distance_function, # 传递距离函数
        'alpha_iter': ALPHA_ITER,
        'threshold_inter': THRESHOLD_INTER,
        'd_disappear': D_DISAPPEAR,
    }

# 在文件底部，我们可以直接获取配置字典
# 在其他文件中，可以通过 `from .SA_config import config` 来导入
config = get_sparse_attn_config()