import os
import glob
import cv2
import numpy as np

def calculate_psnr(img1, img2):
    """计算两张图像之间的 PSNR"""
    # img1 和 img2 是 [H, W, 3] 的 uint8 或 float 数组
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def load_video(path):
    """加载视频为 numpy 数组 [T, H, W, 3]"""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) # [T, H, W, 3]

def main():
    # --- 配置路径 ---
    folder_real = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/dense"
    folder_gen = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/sfcdc"
    # folder_gen = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/SVG2"
    # folder_gen = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/sfcdc_N512G5QBKB_2"
    
    # 获取文件列表
    files_real_map = {os.path.basename(f): f for f in glob.glob(os.path.join(folder_real, "*.mp4"))}
    files_gen_map = {os.path.basename(f): f for f in glob.glob(os.path.join(folder_gen, "*.mp4"))}
    
    # 取交集：只计算两边都有的文件名
    common_names = sorted(list(set(files_real_map.keys()) & set(files_gen_map.keys())))
    num_videos = len(common_names)
    
    if num_videos == 0:
        print("No matching video pairs found based on filenames!")
        return

    print(f"Compute PSNR with {num_videos} pairs of matched videos...")

    all_video_psnrs = []

    for i, name in enumerate(common_names):
        vid_real = load_video(files_real_map[name])
        vid_gen = load_video(files_gen_map[name])
        
        # 确保帧数对齐
        T = min(len(vid_real), len(vid_gen))
        
        frame_psnrs = []
        for t in range(T):
            p = calculate_psnr(vid_real[t], vid_gen[t])
            frame_psnrs.append(p)
        
        avg_v_psnr = np.mean(frame_psnrs)
        all_video_psnrs.append(avg_v_psnr)
        print(f"Video {i+1}/{num_videos} [{name}]: PSNR = {avg_v_psnr:.2f} dB")

    overall_psnr = np.mean(all_video_psnrs)
    print("\n" + "="*40)
    print(f"Avg PSNR: {overall_psnr:.4f} dB")
    print("="*40)

if __name__ == "__main__":
    main()