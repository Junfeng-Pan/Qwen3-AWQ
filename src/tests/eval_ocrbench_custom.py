import torch
import os
import json
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

# 配置
MODEL_PATH = "Qwen3-VL-8B-Instruct-AWQ"
DATASET_PATH = os.path.join("datasets", "OCRBench_dataset")
OUTPUT_FILE = os.path.join("src", "outputs", "OCRBench_eval_results.json")

def evaluate_ocrbench():
    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 2. 加载数据集
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
        if hasattr(dataset, 'keys'):
            # OCRBench 通常在 'test' split 中，但为了健壮性，我们尝试获取第一个 split
            split_keys = list(dataset.keys())
            target_split = 'test' if 'test' in split_keys else ('train' if 'train' in split_keys else split_keys[0])
            print(f"Using dataset split: {target_split}")
            dataset = dataset[target_split]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded. Size: {len(dataset)}")
    
    # 检查数据集列名
    print(f"Features: {dataset.features}")
    
    correct_count = 0
    total_count = 0
    results = []

    # 3. 遍历评测
    print("Starting evaluation...")
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        # OCRBench 数据结构通常是: image (PIL), question (str), answer (str/list), type (str)
        # 具体的列名可能需要适配，这里假设标准字段
        
        image = item.get('image')
        question = item.get('question')
        gt_answer = item.get('answer') # 可能是字符串或列表
        
        if image is None or question is None:
            continue
            
        # 构造对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # 预处理
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False # 贪婪解码用于评测
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 评分逻辑
        # OCRBench 的评分通常比较宽松：不区分大小写，检查包含关系
        is_correct = False
        
        norm_output = output_text.lower().strip()
        
        # 处理 gt_answer 为列表或单个字符串的情况
        # 注意：opencsg/OCRBench 的 answer 字段通常是一个字符串
        possible_answers = []
        if isinstance(gt_answer, list):
            possible_answers = [str(a).lower().strip() for a in gt_answer]
        else:
            possible_answers = [str(gt_answer).lower().strip()]
            
        # 简单判分：如果预测结果包含了正确答案，算对
        # 对于 OCR 任务，这通常是一个合理的基准，特别是对于短答案
        # 更严格的评测可能需要精确匹配
        
        # 针对 OCRBench 官方评测逻辑，这里做一个简单的复现：
        # 如果 output 包含答案，或者答案包含 output (防止生成过多废话)
        for ans in possible_answers:
            if ans in norm_output or norm_output in ans:
                is_correct = True
                break
        
        if is_correct:
            correct_count += 1
            
        total_count += 1
        
        # 记录结果
        result_item = {
            "id": i,
            "question": question,
            "ground_truth": gt_answer,
            "prediction": output_text,
            "is_correct": is_correct
        }
        results.append(result_item)
        
        # 实时打印每 100 条的准确率
        if (i + 1) % 100 == 0:
            print(f"Progress {i+1}/{len(dataset)} | Current Acc: {correct_count/total_count:.2%}")

    # 4. 最终统计
    final_acc = correct_count / total_count if total_count > 0 else 0
    print(f"\n{'='*30}")
    print(f"Final Evaluation Result")
    print(f"Total Samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {final_acc:.2%}")
    print(f"{'='*30}")

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate_ocrbench()
