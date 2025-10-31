import torch
import numpy as np
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_i3d import InceptionI3d
import os

# --- 配置参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 更新为下载的Kinetics-400权重路径（请替换为你的实际路径）
I3D_WEIGHTS_PATH = "./models/model_rgb.pth"
# --- 核心功能复现 ---

def preprocess_torch(videos, target_resolution=(224, 224)):
    """
    使用 PyTorch 复现 FVD 的预处理步骤。
    Args:
        videos: <torch.Tensor>[batch_size, num_frames, height, width, 3]，像素值范围 0-255
    Returns:
        <torch.Tensor>[batch_size, 3, num_frames, height, width]，像素值范围 -1 到 1
    """
    b, t, h, w, c = videos.shape
    
    # PyTorch 的插值函数需要 (B, C, H, W) 格式
    videos = videos.permute(0, 1, 4, 2, 3).contiguous().float()
    videos = videos.view(b * t, c, h, w)
    
    # 使用双线性插值调整大小
    resized_videos = F.interpolate(
        videos,
        size=target_resolution,
        mode='bilinear',
        align_corners=False)
    
    # 调整回视频格式
    resized_videos = resized_videos.view(b, t, c, target_resolution[0], target_resolution[1])
    
    # 像素值从 [0, 255] 归一化到 [-1, 1]
    scaled_videos = 2.0 * (resized_videos / 255.0) - 1.0
    
    # 调整为 I3D 模型期望的输入格式 [B, C, T, H, W]
    return scaled_videos.permute(0, 2, 1, 3, 4)

def get_i3d_model(weights_path=I3D_WEIGHTS_PATH):
    print(f"正在从 '{weights_path}' 加载 I3D 模型...")
    num_classes = 400
    i3d = InceptionI3d(num_classes, in_channels=3)
    state = torch.load(weights_path, map_location=DEVICE)
    if 'state_dict' in state:
        state = state['state_dict']
    
    # 映射键名：将小写键转换为大写键
    new_state = {}
    for k, v in state.items():
        # 替换小写为大写，并调整BatchNorm键
        new_k = k.replace('conv3d_', 'Conv3d_').replace('mixed_', 'Mixed_').replace('batch3d', 'bn').replace('branch_', 'b').replace('.', '.')
        # 特殊处理logits层
        if 'conv3d_0c_1x1' in new_k:
            new_k = new_k.replace('conv3d_0c_1x1', 'logits')
        new_state[new_k] = v
    
    i3d.load_state_dict(new_state, strict=False)  # 使用strict=False以忽略未匹配键
    i3d.to(DEVICE)
    i3d.eval()
    print("I3D 模型加载完毕。")
    return i3d

@torch.no_grad()  # 确保不计算梯度
def calculate_i3d_embeddings(videos, i3d_model):
    """
    为一批视频计算 I3D 特征向量。
    Args:
        videos: <torch.Tensor>[batch_size, 3, num_frames, 224, 224]，经过预处理的视频
        i3d_model: 加载好的 I3D 模型
    Returns:
        <np.ndarray>[batch_size, embedding_size]
    """
    videos = videos.to(DEVICE)
    embeddings = i3d_model.extract_features(videos)
    embeddings = embeddings.mean(dim=2)  # 匹配 TensorFlow Mean:0
    embeddings = embeddings.squeeze(-1).squeeze(-1)
    return embeddings.cpu().numpy()

def calculate_fvd_from_activations(real_activations, generated_activations):
    """
    根据两组特征向量计算 FVD 分数。
    Args:
        real_activations: <np.ndarray>[num_samples, embedding_size]
        generated_activations: <np.ndarray>[num_samples, embedding_size]
    Returns:
        FVD 分数 (float)
    """
    # 计算均值和协方差
    real_activations = real_activations.astype(np.float64)
    generated_activations = generated_activations.astype(np.float64)
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False, dtype=np.float64)
    mu_gen = np.mean(generated_activations, axis=0)
    sigma_gen = np.cov(generated_activations, rowvar=False, dtype=np.float64)
    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

def main():
    """
    复现 TF 官方 example.py 的逻辑：计算全黑视频和全白视频之间的 FVD。
    """
    print(f"使用设备: {DEVICE}")
    
    # 1. 准备数据 (与 TF example.py 一致)
    NUMBER_OF_VIDEOS = 16
    VIDEO_LENGTH = 15
    
    # 创建全黑视频张量
    first_set_of_videos = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3, dtype=torch.uint8)
    
    # 创建全白视频张量
    second_set_of_videos = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3, dtype=torch.uint8) * 255
    
    # 2. 加载 I3D 模型
    i3d_model = get_i3d_model()
    
    # 3. 预处理视频
    print("正在预处理视频...")
    preprocessed_videos1 = preprocess_torch(first_set_of_videos)
    preprocessed_videos2 = preprocess_torch(second_set_of_videos)
    
    # 4. 提取特征
    print("正在提取第一组视频的特征...")
    embeddings1 = calculate_i3d_embeddings(preprocessed_videos1, i3d_model)
    
    print("正在提取第二组视频的特征...")
    embeddings2 = calculate_i3d_embeddings(preprocessed_videos2, i3d_model)
    
    # 5. 计算 FVD
    print("正在计算 FVD 分数...")
    fvd_score = calculate_fvd_from_activations(embeddings1, embeddings2)
    
    print("-" * 30)
    print(f"计算得到的 FVD (PyTorch) 是: {fvd_score:.2f}")
    print("(原始 TF 代码的参考值约为 131)")
    print("-" * 30)

if __name__ == "__main__":
    main()