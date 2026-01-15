import os
import glob
import cv2
import numpy as np
import torch
import scipy.linalg as linalg

# --- 配置参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
I3D_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "./models/i3d_torchscript.pt")

def load_full_videos(paths, video_length):
    """
    直接加载整格视频（不裁切），统一时间步长。
    """
    all_videos = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            # 保持全画幅，模型后续会自动 resize 到 224x224
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(frames) == 0: continue
        
        if len(frames) >= video_length:
            idx = np.linspace(0, len(frames) - 1, num=video_length, dtype=int)
            sampled = [frames[i] for i in idx]
        else:
            sampled = frames + [frames[-1]] * (video_length - len(frames))
        
        all_videos.append(np.stack(sampled, axis=0)) # [T, H, W, 3]
            
    if len(all_videos) == 0:
        raise ValueError("未加载到视频内容。")
        
    return np.stack(all_videos, axis=0).astype(np.uint8)

def get_file_list(folder, ext_list=("mp4", "avi", "mov", "mkv")):
    files = []
    for ext in ext_list:
        files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
    files.sort()
    return files

@torch.no_grad()
def calculate_i3d_embeddings(videos, i3d_model, batch_size=8):
    i3d_model.eval()
    all_embeddings = []
    for i in range(0, len(videos), batch_size):
        batch = videos[i : i + batch_size]
        batch_tensor = torch.from_numpy(batch).permute(0, 4, 1, 2, 3).to(DEVICE).float().contiguous()
        # 模型内置 resize=True 会把 1280x704 压缩到 224x224
        emb = i3d_model(batch_tensor, rescale=True, resize=True, return_features=True)
        all_embeddings.append(emb.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

def calculate_fvd(real_videos, generated_videos, i3d_model):
    print(f"提取特征中... 样本量: {len(real_videos)}")
    real_feats = calculate_i3d_embeddings(real_videos, i3d_model)
    gen_feats = calculate_i3d_embeddings(generated_videos, i3d_model)
    
    mu_real = np.mean(real_feats, axis=0)
    mu_gen = np.mean(gen_feats, axis=0)
    
    # 因为只有8个样本，协方差矩阵极度不稳定，正则化必不可少
    def stable_cov(feats, reg=1e-5):
        cov = np.cov(feats, rowvar=False)
        return cov + reg * np.eye(cov.shape[0])

    sigma_real = stable_cov(real_feats)
    sigma_gen = stable_cov(gen_feats)
    
    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

def main():
    folder_a = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/dense"
    folder_b = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/sfcdc"
    VIDEO_LENGTH = 53
    TARGET_COUNT = 16  # 你希望复用到的数量
    
    print(f"Working on {DEVICE}")
    i3d_model = torch.jit.load(I3D_WEIGHTS_PATH).to(DEVICE).eval()
    
    files_a = get_file_list(folder_a)[:8]
    files_b = get_file_list(folder_b)[:8]
    
    # 1. 加载原始 8 个视频
    print(f"Loading original videos...")
    vids_a_raw = load_full_videos(files_a, VIDEO_LENGTH)
    vids_b_raw = load_full_videos(files_b, VIDEO_LENGTH)
    
    # 2. 复用逻辑：如果数量不足，进行循环复制
    def repeat_to_target(vids, target):
        current_count = vids.shape[0]
        if current_count >= target:
            return vids[:target]
        repeats = (target // current_count) + 1
        extended = np.tile(vids, (repeats, 1, 1, 1, 1))
        return extended[:target]

    vids_a = repeat_to_target(vids_a_raw, TARGET_COUNT)
    vids_b = repeat_to_target(vids_b_raw, TARGET_COUNT)
    
    fvd_score = calculate_fvd(vids_a, vids_b, i3d_model)
    
    print("\n" + "="*40)
    print(f"Final FVD (Full Frame): {fvd_score:.4f}")
    print(f"Note: Based on {vids_a_raw.shape[0]} unique videos repeated to {TARGET_COUNT}")
    print("="*40)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



if __name__ == "__main__":
    main()