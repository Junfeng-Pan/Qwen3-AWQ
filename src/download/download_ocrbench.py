import os
from datasets import load_dataset

# 数据集 ID
DATASET_ID = "opencsg/OCRBench"
# 本地保存路径
SAVE_DIR = os.path.join("datasets", "OCRBench_dataset")

def download_ocrbench():
    print(f"开始下载数据集: {DATASET_ID}")
    try:
        # 加载数据集
        # OCRBench 通常只有一个 split，或者我们取 train
        ds = load_dataset(DATASET_ID)
        
        print(f"数据集信息: {ds}")
        
        # 保存到本地
        print(f"正在保存到本地目录: {SAVE_DIR} ...")
        ds.save_to_disk(SAVE_DIR)
        print(f"保存成功！路径: {os.path.abspath(SAVE_DIR)}")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请检查网络连接或是否需要配置 HF 镜像代理。")

if __name__ == "__main__":
    download_ocrbench()
