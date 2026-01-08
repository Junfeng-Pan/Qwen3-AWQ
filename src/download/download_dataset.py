import os
from datasets import load_dataset

# 数据集配置
DATASET_ID = "neuralmagic/calibration"
SUBSETS = ["LLM", "VLM"]
BASE_SAVE_DIR = os.path.join("datasets", "calibration_dataset")

print(f"准备下载数据集: {DATASET_ID}")

for subset in SUBSETS:
    save_dir = os.path.join(BASE_SAVE_DIR, subset)
    print(f"\n正在处理子集: {subset}...")
    
    try:
        # 加载数据集
        print(f"正在从 Hugging Face 下载 {subset}...")
        ds = load_dataset(DATASET_ID, name=subset)
        
        # 保存到本地目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        print(f"下载完成，正在保存到本地目录: {save_dir}")
        ds.save_to_disk(save_dir)
        print(f"子集 {subset} 已成功保存！")

    except Exception as e:
        print(f"\n下载或保存 {subset} 失败: {e}")
        print("\n提示: 如果网络连接困难，请确保已配置代理或尝试设置环境变量:")
        print("export HF_ENDPOINT=https://hf-mirror.com")

print("\n所有指定的校准数据集已处理完成！")
print(f"在量化脚本中，你可以根据需要选择加载：")
print(f"VLM 数据 (推荐用于 Qwen-VL): ds = load_from_disk('{os.path.join(BASE_SAVE_DIR, 'VLM')}')['train']")
print(f"LLM 数据: ds = load_from_disk('{os.path.join(BASE_SAVE_DIR, 'LLM')}')['train']")