import os
import io
import torch
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

# 1. 设置模型路径和输出路径
# 假设模型已经下载到当前目录下的 Qwen2.5-VL-7B-Instruct
MODEL_ID = "Qwen2.5-VL-7B-Instruct" 
SAVE_DIR = f"{MODEL_ID}-AWQ"
# 使用 VLM 子集
DATASET_PATH = os.path.join("datasets", "calibration_dataset", "VLM") 

print(f"Loading model from {MODEL_ID}...")

# 2. 加载模型和处理器
try:
    # Qwen2.5-VL 通常使用 Qwen2_5_VLForConditionalGeneration 或 AutoModelForVision2Seq
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
except OSError:
    print(f"Error: Model not found at {MODEL_ID}. Please run the download script first.")
    exit(1)

# 3. 准备校准数据集
print(f"Loading local calibration dataset from {DATASET_PATH}...")
try:
    # 从本地磁盘加载之前下载的数据集
    ds = load_from_disk(DATASET_PATH)
    # 针对 neuralmagic/calibration 数据集，通常我们需要其中的 train 分片
    if isinstance(ds, dict):
        ds = ds["train"]
    
    # 减少样本数量以加快速度，VLM 数据较大
    NUM_CALIBRATION_SAMPLES = 128 
    MAX_SEQUENCE_LENGTH = 2048
    
    ds = ds.shuffle(seed=42).select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure you have run 'python src/download/download_dataset.py' successfully.")
    exit(1)

def preprocess_function(example):
    # 处理 VLM 数据集中的 messages
    messages = example["messages"]
    
    # 遍历 messages，将 image bytes 转换为 PIL Image
    new_messages = []
    for msg in messages:
        new_content = []
        for item in msg["content"]:
            if "image" in item:
                # 处理图像数据
                img_data = item.get("image")
                
                # 增加空值检查
                if img_data is None:
                    # print("Warning: Found None image in dataset. Skipping this image item.")
                    continue

                if isinstance(img_data, bytes):
                    try:
                        image = Image.open(io.BytesIO(img_data))
                        # 确保转换为 RGB，防止 RGBA 等导致的问题
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        new_content.append({"type": "image", "image": image})
                    except Exception as e:
                        print(f"Warning: Failed to decode image bytes: {e}. Skipping.")
                        continue
                elif isinstance(img_data, (str, Image.Image)):
                    # 已经是 path 或 PIL Image
                    new_content.append({"type": "image", "image": img_data})
                else:
                    # 其他未知类型，跳过以防报错
                    print(f"Warning: Unknown image type {type(img_data)}. Skipping.")
                    continue
            elif "text" in item:
                new_content.append({"type": "text", "text": item["text"]})
            else:
                # Fallback
                new_content.append(item)
        
        # 只有当 content 不为空时才添加 message
        if new_content:
            new_messages.append({"role": msg["role"], "content": new_content})

    # 如果处理后没有有效消息，返回空或者处理异常（这里简单处理为只要有文本或图片就继续）
    if not new_messages:
        # 返回一个 dummy 数据防止报错，或者让 map 过滤掉（通过后续逻辑）
        # 这里我们构造一个纯文本的 dummy message
        new_messages = [{"role": "user", "content": [{"type": "text", "text": "ignore"}]}]

    # 使用 qwen_vl_utils 处理视觉信息
    try:
        image_inputs, video_inputs = process_vision_info(new_messages)
    except Exception as e:
        print(f"Warning: process_vision_info failed: {e}. Using dummy inputs.")
        image_inputs, video_inputs = None, None
        # Reset to text only
        new_messages = [{"role": "user", "content": [{"type": "text", "text": "ignore"}]}]
    
    # 应用聊天模板
    text = processor.apply_chat_template(
        new_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    
    # Convert tensors to lists for Arrow dataset storage
    result = {
        "input_ids": inputs["input_ids"][0].tolist(),
        "attention_mask": inputs["attention_mask"][0].tolist(),
    }
    
    if "pixel_values" in inputs:
        result["pixel_values"] = inputs["pixel_values"].tolist() # Flattened list
        result["image_grid_thw"] = inputs["image_grid_thw"].tolist()
        
    if "pixel_values_videos" in inputs:
        result["pixel_values_videos"] = inputs["pixel_values_videos"].tolist()
        result["video_grid_thw"] = inputs["video_grid_thw"].tolist()
        
    return result

# Apply preprocessing to the dataset
print("Preprocessing dataset...")
ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)

def data_collator(batch):
    # Batch is a list of dicts (from the processed dataset)
    
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    attention_mask = [torch.tensor(x['attention_mask']) for x in batch]
    
    # Pad text inputs
    from torch.nn.utils.rnn import pad_sequence
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    batch_out = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded
    }
    
    # 处理视觉特征 (concat flattened features)
    if "pixel_values" in batch[0]:
        batch_out["pixel_values"] = torch.cat([torch.tensor(x["pixel_values"]) for x in batch], dim=0)
        batch_out["image_grid_thw"] = torch.cat([torch.tensor(x["image_grid_thw"]) for x in batch], dim=0)
        
    if "pixel_values_videos" in batch[0]:
         batch_out["pixel_values_videos"] = torch.cat([torch.tensor(x["pixel_values_videos"]) for x in batch], dim=0)
         batch_out["video_grid_thw"] = torch.cat([torch.tensor(x["video_grid_thw"]) for x in batch], dim=0)

    return batch_out

# 4. 配置 AWQ 量化配方 (Recipe)
print("Configuring AWQ recipe...")

# Qwen2.5-VL 结构可能与 Qwen3-VL 类似，使用相同的映射策略
# 如果自动推断失败，这些映射将起作用
mappings = [
    AWQMapping(
        smooth_layer="re:.*input_layernorm",
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    AWQMapping(
        smooth_layer="re:.*post_attention_layernorm",
        balance_layers=["re:.*gate_proj", "re:.*up_proj"],
    ),
    AWQMapping(
        smooth_layer="re:.*up_proj",
        balance_layers=["re:.*down_proj"],
    ),
]

recipe = AWQModifier(
    # 忽略视觉编码器部分，通常视觉部分对量化敏感且参数量占比相对较小
    ignore=[
        "re:.*embed_tokens",
        "re:model[.]visual.*", # 忽略视觉塔
        "re:visual.*",
        "lm_head",
        "re:.*q_norm", 
        "re:.*k_norm",
    ],
    mappings=mappings,
    duo_scaling=True,
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 128,
                "strategy": "group",
                "dynamic": False,
                "actorder": None,
                "observer": "mse",
            },
        }
    },
)

# 5. 执行量化
print("Starting quantization (this may take a while)...")
oneshot(
    model=model,
    processor=processor, # 传入 processor
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

# 6. 保存量化模型
print(f"Saving quantized model to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR) # 保存 processor 配置
print("Quantization complete!")
