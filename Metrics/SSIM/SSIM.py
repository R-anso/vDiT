import os
import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    """计算两张图像之间的 SSIM"""
    # SSIM 对彩色图像通常在每个通道分别计算再取平均，或者直接指定 channel_axis
    # data_range=255 表示像素值范围是 0-255
    score, _ = ssim(img1, img2, full=True, channel_axis=2, data_range=255)
    return score

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

def main():
    # --- 配置路径 ---
    folder_real = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/dense"
    folder_gen = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/sfcdc"
    
    # 获取文件列表
    files_real = sorted(glob.glob(os.path.join(folder_real, "*.mp4")))
    files_gen = sorted(glob.glob(os.path.join(folder_gen, "*.mp4")))
    
    num_videos = min(len(files_real), len(files_gen))
    print(f"Compute SSIM with {num_videos} pairs of videos...")

    all_video_ssims = []

    for i in range(num_videos):
        vid_real = load_video(files_real[i])
        vid_gen = load_video(files_gen[i])
        
        # 确保帧数对齐
        T = min(len(vid_real), len(vid_gen))
        
        frame_ssims = []
        for t in range(T):
            # SSIM 计算
            score = calculate_ssim(vid_real[t], vid_gen[t])
            frame_ssims.append(score)
        
        avg_v_ssim = np.mean(frame_ssims)
        all_video_ssims.append(avg_v_ssim)
        print(f"Video {i+1}/{num_videos} [{os.path.basename(files_gen[i])}]: SSIM = {avg_v_ssim:.4f}")

    overall_ssim = np.mean(all_video_ssims)
    print("\n" + "="*40)
    print(f"Avg SSIM: {overall_ssim:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()