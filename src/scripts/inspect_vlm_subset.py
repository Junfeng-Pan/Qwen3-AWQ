from datasets import load_dataset

dataset_id = "neuralmagic/calibration"
subset = "VLM"

try:
    print(f"Inspecting {dataset_id} subset {subset}...")
    ds = load_dataset(dataset_id, name=subset, split="train", streaming=True)
    
    # Get first example
    example = next(iter(ds))
    print("Keys:", example.keys())
    print("Example content (truncated):", str(example)[:500])
    
    # Check if image is PIL or something else
    if "image" in example:
        print(f"Image type: {type(example['image'])}")
        
except Exception as e:
    print(f"Error: {e}")
