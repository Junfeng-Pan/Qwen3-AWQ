from transformers import AutoModelForVision2Seq
import torch

model_path = "Qwen3-VL-8B-Instruct-AWQ"

print(f"Loading model from {model_path}...")
try:
    # 尝试加载模型
    model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto")
    print("Model loaded successfully.")
    
    print("\n--- Model Structure Inspection ---")
    
    # 检查文本部分的某一层
    text_layer = model.model.language_model.layers[0]
    print(f"Text Layer 0 Structure: {text_layer}")
    
    # 检查具体的 Linear 层，看是否为量化后的类型
    # 在 transformers 中，加载 safetensors 量化模型后，权重通常是 float/bfloat16 (dequantized on load if not using specific kernel) 
    # 或者如果是 GPTQ/AWQ 集成，可能会显示为 QuantizedLinear。
    # llmcompressor 输出的 compressed-tensors 格式，如果是用 transformers 直接加载，
    # 且没有安装特定的 kernel 插件，可能会显示为普通 Linear 但权重被 pack 过？
    # 不，通常 transformers 会识别 quantization_config 并尝试加载对应的 kernel。
    # 让我们看看 q_proj 是什么类型。
    q_proj = text_layer.self_attn.q_proj
    print(f"\nType of q_proj: {type(q_proj)}")
    print(f"q_proj details: {q_proj}")

    # 检查视觉部分的某一层，应该保持原样
    visual_block = model.model.visual.blocks[0]
    print(f"\nVisual Block 0 Attention qkv type: {type(visual_block.attn.qkv)}")
    
except Exception as e:
    print(f"Error loading or inspecting model: {e}")
