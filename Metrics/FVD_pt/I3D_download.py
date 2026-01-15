import os
from huggingface_hub import hf_hub_download

# 设置保存路径
save_dir = os.path.abspath("src/FVD_pt/models")
os.makedirs(save_dir, exist_ok=True)

# 下载模型
model_path = hf_hub_download(
    repo_id="flateon/FVD-I3D-torchscript",
    filename="i3d_torchscript.pt",
    local_dir=save_dir,
    local_dir_use_symlinks=False
)
print(f"模型下载完成，保存路径为: {model_path}")