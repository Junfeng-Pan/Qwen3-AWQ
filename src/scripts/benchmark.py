import time
import torch
import argparse
import statistics
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

def benchmark_model(model_path, num_iters=5, max_new_tokens=128):
    print(f"{'='*20} Benchmarking: {model_path} {'='*20}")
    
    # 1. 加载模型和处理器
    print("Loading model...")
    start_load = time.time()
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    # 2. 准备输入 (使用一张占位图或实际图片)
    # 使用之前测试过的稳定链接
    image_url = "https://qcloud.dpfile.com/pc/du_9iEcvKMPi4UqsEfm7-ptrvLB-tvoSGEWtPWN33mLb496vDQVmtOj-cPABMfP4.jpg"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Describe this image in detail, covering colors, objects, and atmosphere."}, # 英文 prompt 往往能触发更稳定的长回复，利于测速
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 3. 预热 (Warmup)
    # 让 GPU 和 CUDA kernel 进入工作状态，第一次推理通常较慢，不计入统计
    print("Warming up...")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # 清空显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 4. 循环测试
    print(f"Running {num_iters} iterations (max_new_tokens={max_new_tokens})...")
    latencies = []
    token_counts = []
    
    for i in range(num_iters):
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                min_new_tokens=max_new_tokens, # 强制生成足够长
                do_sample=False # 贪婪解码，保证确定性
            )
        end_time = time.time()
        
        # 计算生成的 token 数量 (输出总长度 - 输入长度)
        generated_tokens = output.shape[1] - inputs.input_ids.shape[1]
        
        latency = end_time - start_time
        latencies.append(latency)
        token_counts.append(generated_tokens)
        
        tps = generated_tokens / latency
        print(f"Iter {i+1}: {latency:.2f}s | {generated_tokens} tokens | {tps:.2f} tokens/sec")

    # 5. 统计结果
    avg_latency = statistics.mean(latencies)
    avg_tps = sum(token_counts) / sum(latencies)
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB

    print(f"\n{'-'*20} Results: {model_path} {'-'*20}")
    print(f"Average TPS (Tokens/sec): {avg_tps:.2f}")
    print(f"Average Latency:          {avg_latency:.2f} s")
    print(f"Peak Memory Usage:        {max_memory:.2f} GB")
    print(f"{ '='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--iters", type=int, default=5, help="Number of test iterations")
    parser.add_argument("--tokens", type=int, default=128, help="Max new tokens to generate")
    args = parser.parse_args()

    benchmark_model(args.model_path, args.iters, args.tokens)
