import os
from datasets import load_dataset

# 数据集配置
DATASET_ID = "neuralmagic/calibration"
DATASET_NAME = "LLM" # 对应示例代码中的子集
SAVE_DIR = os.path.join("datasets", "calibration_dataset")

print(f"准备下载数据集: {DATASET_ID} (subset: {DATASET_NAME})")

try:
    # 加载数据集
    # 如果环境中有代理，load_dataset 会自动使用
    print("正在从 Hugging Face 下载...")
    ds = load_dataset(DATASET_ID, name=DATASET_NAME)
    
    # 保存到本地目录
    print(f"下载完成，正在保存到本地目录: {SAVE_DIR}")
    ds.save_to_disk(SAVE_DIR)
    
    print("\n数据集已成功保存！")
    print(f"在量化脚本中，你可以使用以下方式加载本地数据集：")
    print(f"ds = load_from_disk('{SAVE_DIR}')['train']")

except Exception as e:
    print(f"\n下载或保存失败: {e}")
    print("\n提示: 如果网络连接困难，请确保已配置代理或尝试设置环境变量:")
    print("export HF_ENDPOINT=https://hf-mirror.com")
