import os
import glob
import cv2
import numpy as np
import torch
import lpips

# --- 配置参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_video(path):
    """加载视频为 numpy 数组 [T, H, W, 3]"""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)

def preprocess_for_lpips(frame):
    """
    将 numpy [H, W, 3] uint8 转换为 torch [1, 3, H, W] float
    像素范围从 [0, 255] 映射到 [-1, 1] (LPIPS 期望的范围)
    """
    frame = frame.astype(np.float32) / 255.0
    frame = 2.0 * frame - 1.0
    frame = frame.transpose(2, 0, 1) # HWC -> CHW
    frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
    return frame

def main():
    # --- 配置路径 ---
    folder_real = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/dense"
    folder_gen = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/sfcdc"
    # folder_gen = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/SVG2"
    # folder_gen = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/
    
    # 1. 初始化 LPIPS 模型 (net='alex' 速度最快且最常用，'vgg' 更精准但更慢)
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
    loss_fn.eval()
    
    # 获取文件列表
    files_real_map = {os.path.basename(f): f for f in glob.glob(os.path.join(folder_real, "*.mp4"))}
    files_gen_map = {os.path.basename(f): f for f in glob.glob(os.path.join(folder_gen, "*.mp4"))}
    
    # 取交集
    common_names = sorted(list(set(files_real_map.keys()) & set(files_gen_map.keys())))
    num_videos = len(common_names)
    
    if num_videos == 0:
        print("No matching video pairs found based on filenames!")
        return

    print(f"Compute LPIPS (AlexNet) with {num_videos} pairs of matched videos on {DEVICE}...")

    all_video_lpips = []

    with torch.no_grad():
        for i, name in enumerate(common_names):
            vid_real = load_video(files_real_map[name])
            vid_gen = load_video(files_gen_map[name])
            
            # 确保帧数对齐
            T = min(len(vid_real), len(vid_gen))
            
            frame_lpips = []
            for t in range(T):
                # 预处理
                img_real = preprocess_for_lpips(vid_real[t])
                img_gen = preprocess_for_lpips(vid_gen[t])
                
                # 计算 LPIPS (值越小越相似)
                dist = loss_fn(img_real, img_gen)
                frame_lpips.append(dist.item())
            
            avg_v_lpips = np.mean(frame_lpips)
            all_video_lpips.append(avg_v_lpips)
            print(f"Video {i+1}/{num_videos} [{name}]: LPIPS = {avg_v_lpips:.4f}")

    overall_lpips = np.mean(all_video_lpips)
    print("\n" + "="*40)
    print(f"Avg LPIPS: {overall_lpips:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()