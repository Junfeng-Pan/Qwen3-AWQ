import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型路径
MODEL_PATH = "Qwen3-VL-8B-Instruct-AWQ"

# 用户提供的图片链接 (从搜索结果 URL 中提取的直接图片地址)
# 原 URL 是百度图片详情页，直接传入可能会导致解析错误，因此提取了其中的 objurl 参数
IMAGE_URL = "https://qcloud.dpfile.com/pc/du_9iEcvKMPi4UqsEfm7-ptrvLB-tvoSGEWtPWN33mLb496vDQVmtOj-cPABMfP4.jpg"

def run_inference():
    print(f"Loading model from {MODEL_PATH}...")
    # 加载模型
    # 注意：由于安装了 llmcompressor，它会注册相关处理逻辑，使得 transformers 可以加载 compressed-tensors 格式
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print("Model loaded successfully.")
    print(f"Processing image: {IMAGE_URL}")
    
    # 构造输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_URL,
                },
                {"type": "text", "text": "请详细描述这张图片的内容。"},
            ],
        }
    ]
    
    # 准备推理输入
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
    
    # 将输入移动到 GPU
    inputs = inputs.to("cuda")
    
    # 生成回答
    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False  # 使用贪婪解码以获得确定性结果，也可以开启 sample
        )
        
    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n" + "="*20 + " 模型回答 " + "="*20)
    print(output_text[0])
    print("="*50)

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"\n发生错误: {e}")
