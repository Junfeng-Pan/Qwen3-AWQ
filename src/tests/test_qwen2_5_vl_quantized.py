import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import os

# 1. 配置路径
MODEL_PATH = "Qwen2.5-VL-7B-Instruct-AWQ"
IMAGE_URL = "https://qcloud.dpfile.com/pc/du_9iEcvKMPi4UqsEfm7-ptrvLB-tvoSGEWtPWN33mLb496vDQVmtOj-cPABMfP4.jpg"

print(f"正在加载量化模型: {MODEL_PATH}...")

try:
    # 加载模型
    # 注意：量化模型通常需要 trust_remote_code=True
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("模型加载成功！")

    # 2. 准备推理数据
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_URL,
                },
                {"type": "text", "text": "请描述这张图片中的内容。"},
            ],
        }
    ]

    # 准备输入
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
    inputs = inputs.to(model.device)

    # 3. 执行推理
    print("正在生成回答...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    print("\n" + "="*20 + " 模型回答 " + "="*20)
    print(output_text[0])
    print("="*50)

except Exception as e:
    print(f"测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
