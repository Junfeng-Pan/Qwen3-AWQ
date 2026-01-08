from transformers import AutoModelForVision2Seq
import torch

model_id = "Qwen2.5-VL-7B-Instruct"

print(f"Loading model {model_id} structure...")
try:
    # Load on meta device to save memory, just to check structure
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        trust_remote_code=True,
        device_map="meta" 
    )
    
    print("\nTop level modules:")
    for name, module in model.named_children():
        print(f"- {name}")
        
    print("\nVisual module children (first level):")
    if hasattr(model, "visual"):
        for name, module in model.visual.named_children():
            print(f"- visual.{name} ({type(module).__name__})")
            
except Exception as e:
    print(f"Error: {e}")
