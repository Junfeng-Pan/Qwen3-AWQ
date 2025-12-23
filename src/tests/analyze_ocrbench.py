import json
import os
from collections import Counter

# 结果文件路径
RESULT_FILE = os.path.join("src", "outputs", "OCRBench_eval_results.json")

def analyze_results():
    if not os.path.exists(RESULT_FILE):
        print(f"错误: 找不到结果文件 {RESULT_FILE}。请先运行评测脚本。")
        return

    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total = len(results)
    correct = sum(1 for item in results if item['is_correct'])
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\n" + "="*60)
    print(f"{ 'OCRBench 评测分析报告':^56}")
    print("="*60)

    # --- 新增：分段准确率统计 ---
    print(f"\n[ 分段统计 (每100条) ]")
    print(f"{ 'Range':<15} | {'Correct':<10} | {'Batch Accuracy'}")
    print("-" * 45)
    
    batch_size = 100
    for i in range(0, total, batch_size):
        batch_items = results[i : i + batch_size]
        batch_total = len(batch_items)
        batch_correct = sum(1 for item in batch_items if item['is_correct'])
        batch_acc = (batch_correct / batch_total) * 100 if batch_total > 0 else 0
        
        range_str = f"{i+1} - {min(i+batch_size, total)}"
        print(f"{range_str:<15} | {batch_correct}/{batch_total:<9} | {batch_acc:.2f}%")
    print("-" * 45)
    # ---------------------------

    print(f"\n[ 总体统计 ]")
    print(f"总样本数: {total}")
    print(f"正确数量: {correct}")
    print(f"总体准确率: {accuracy:.2f}%")
    print("="*60)

    # 抽取部分样例
    print(f"\n[ 成功样例展示 (Correct Samples) ]")
    correct_samples = [item for item in results if item['is_correct']][:3]
    for i, item in enumerate(correct_samples):
        print(f"样例 {i+1}:")
        print(f"  问题: {item['question']}")
        print(f"  预测: {item['prediction']}")
        print(f"  标准答案: {item['ground_truth']}")
        print("-" * 30)

    print(f"\n[ 失败样例展示 (Incorrect Samples) ]")
    incorrect_samples = [item for item in results if not item['is_correct']][:3]
    for i, item in enumerate(incorrect_samples):
        print(f"样例 {i+1}:")
        print(f"  问题: {item['question']}")
        print(f"  预测: {item['prediction']}")
        print(f"  标准答案: {item['ground_truth']}")
        print("-" * 30)

    print("\n分析完毕。")

if __name__ == "__main__":
    analyze_results()