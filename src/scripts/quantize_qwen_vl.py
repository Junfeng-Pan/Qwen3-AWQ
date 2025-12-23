import torch
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForVision2Seq

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

# 1. 设置模型路径和输出路径
MODEL_ID = "Qwen3-VL-8B-Instruct" 
SAVE_DIR = f"{MODEL_ID}-AWQ"
DATASET_PATH = os.path.join("datasets", "calibration_dataset") # 本地数据集路径

print(f"Loading model from {MODEL_ID}...")

# 2. 加载模型和处理器
try:
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
    
    NUM_CALIBRATION_SAMPLES = 128
    MAX_SEQUENCE_LENGTH = 2048
    
    ds = ds.shuffle(seed=42).select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure you have run 'python src/scripts/download_dataset.py' successfully.")
    exit(1)

def preprocess_function(example):
    # neuralmagic/calibration 数据集结构通常包含一个 'messages' 字段或 'text' 字段
    if "messages" in example:
        messages = example["messages"]
    elif "text" in example:
        messages = [{"role": "user", "content": [{"type": "text", "text": example["text"]}]}]
    else:
        # 备选方案：尝试获取所有字符串内容
        content = str(example)
        messages = [{"role": "user", "content": [{"type": "text", "text": content}]}]

    # 某些数据集的 messages 已经是格式化好的，我们需要确保它符合 Qwen-VL 的格式要求
    # 如果 content 是纯文本字符串，将其转换为列表格式
    for msg in messages:
        if isinstance(msg["content"], str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]

    # 应用聊天模板
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return processor(
        text=[text],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

def data_collator(batch):
    processed_batch = [preprocess_function(item) for item in batch]
    
    return {
        "input_ids": torch.cat([x['input_ids'] for x in processed_batch], dim=0),
        "attention_mask": torch.cat([x['attention_mask'] for x in processed_batch], dim=0),
    }

# 4. 配置 AWQ 量化配方 (Recipe)
print("Configuring AWQ recipe...")

# 手动指定层级映射，解决自动推断在 Vision2Seq 模型上的问题
mappings = [
    AWQMapping(
        smooth_layer="re:.*input_layernorm",
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    # 移除 v_proj -> o_proj 映射，因为它们之间通常隔着 Attention 算子，且 Qwen3VL 有 q_norm/k_norm
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



    # 注意：不要忽略 input_layernorm 和 post_attention_layernorm，因为 AWQ 需要修改它们的缩放因子



    ignore=[



        "re:.*embed_tokens",



        "re:model[.]visual.*", # 关键：忽略视觉塔



        "re:visual.*",



        "lm_head",



        "re:.*q_norm", # 忽略 attention 内部的 norm



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

                "group_size": 128, # 使用更通用的 128，也可以改为 32 以获得更高精度

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
