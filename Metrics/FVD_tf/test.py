from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import frechet_video_distance as fvd

# Number of videos must be divisible by 16 (or按你的 FVD 实现要求调整)
NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 53

def load_videos_from_paths(paths, video_length):
  """Load list of video files -> numpy array [N, T, H, W, 3], dtype=uint8.
     - If a video has fewer frames, pad with last frame.
     - If more, sample frames uniformly to video_length.
  """
  videos = []
  for p in paths:
    cap = cv2.VideoCapture(p)
    frames = []
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      # OpenCV gives BGR, convert to RGB
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frames.append(frame)
    cap.release()
    if len(frames) == 0:
      # skip empty file
      continue
    # sample or pad to desired length
    if len(frames) >= video_length:
      # uniform sampling
      idx = np.linspace(0, len(frames) - 1, num=video_length, dtype=int)
      sel = [frames[i] for i in idx]
    else:
      # pad by repeating last frame
      pad_count = video_length - len(frames)
      sel = frames + [frames[-1]] * pad_count
    videos.append(np.stack(sel, axis=0).astype(np.uint8))  # [T,H,W,3]
  if len(videos) == 0:
    raise ValueError("No videos loaded from paths.")
  # stack to [N,T,H,W,3]
  return np.stack(videos, axis=0)

def get_file_list(folder, ext_list=("mp4","avi","mov","mkv")):
  files = []
  for ext in ext_list:
    files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
  files.sort()
  return files

def main(argv):
  del argv
  # 指定两个目录或两个文件列表
  folder_a = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/dense"
  folder_b = "/login_home/shuchang/Desktop/Wan2.2/tests/videos/F53/sfcdc"

  files_a = get_file_list(folder_a)[:NUMBER_OF_VIDEOS]
  files_b = get_file_list(folder_b)[:NUMBER_OF_VIDEOS]

  # 如果你只有一组视频想和空白比较，可以用 zeros/ones 按需构造
  vids_a = load_videos_from_paths(files_a, VIDEO_LENGTH)  # [N,T,H,W,3]
  vids_b = load_videos_from_paths(files_b, VIDEO_LENGTH)

  # 如果数量不足 NUMBER_OF_VIDEOS，可以 pad 或报错
  if vids_a.shape[0] != NUMBER_OF_VIDEOS or vids_b.shape[0] != NUMBER_OF_VIDEOS:
    raise ValueError("请保证每组视频数量等于 NUMBER_OF_VIDEOS 或调整参数。")

  with tf.Graph().as_default():
    # 转为 tf Tensor（uint8，值在 0-255），preprocess 会转换为 float 和 resize
    first_set_of_videos = tf.constant(vids_a, dtype=tf.uint8)
    second_set_of_videos = tf.constant(vids_b, dtype=tf.uint8)

    result = fvd.calculate_fvd(
        fvd.create_id3_embedding(fvd.preprocess(first_set_of_videos, (224, 224))),
        fvd.create_id3_embedding(fvd.preprocess(second_set_of_videos, (224, 224))))

    with tf.Session() as sess:
      # 初始化 hub/KerasLayer 创建的变量和表
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      # 如果 result 是 Tensor，sess.run，否则直接打印数值
      if isinstance(result, tf.Tensor):
        value = sess.run(result)
      else:
        value = result
      print("FVD is: %.2f." % value)

if __name__ == "__main__":
  tf.app.run(main)