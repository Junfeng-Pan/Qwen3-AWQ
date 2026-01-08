import os
import sys

try:
    from modelscope import snapshot_download
except ImportError:
    print("正在安装 modelscope...")
    os.system(f"{sys.executable} -m pip install modelscope")
    from modelscope import snapshot_download

# 模型 ID: Qwen2.5-VL-7B-Instruct
model_id = "qwen/Qwen2.5-VL-7B-Instruct"

# 下载路径：当前目录下的 Qwen2.5-VL-7B-Instruct 文件夹
current_working_dir = os.getcwd()
local_dir = os.path.join(current_working_dir, "Qwen2.5-VL-7B-Instruct")

print(f"准备从魔塔社区下载模型: {model_id}")
print(f"保存路径: {local_dir}")

try:
    snapshot_download(
        model_id, 
        local_dir=local_dir,
        revision='master'
    )
    print(f"\n模型已成功下载到: {local_dir}")
except Exception as e:
    print(f"\n下载出错: {e}")
    print("请检查网络连接或模型 ID 是否正确。")
