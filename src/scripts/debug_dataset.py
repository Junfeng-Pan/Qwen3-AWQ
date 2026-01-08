import os
from datasets import load_from_disk

DATASET_PATH = os.path.join("datasets", "calibration_dataset", "VLM")

print(f"Loading dataset from {DATASET_PATH}...")
try:
    ds = load_from_disk(DATASET_PATH)
    print(f"Type of ds: {type(ds)}")
    
    if hasattr(ds, "keys"):
        print(f"Keys: {ds.keys()}")
    
    if isinstance(ds, dict):
        if "train" in ds:
            ds = ds["train"]
            print("Selected 'train' split.")
        else:
            print("No 'train' key found. Using default.")
            
    print(f"Dataset length: {len(ds)}")
    
    if len(ds) > 0:
        print("First example keys:", ds[0].keys())
        
    # Simulate the shuffle and select
    NUM_CALIBRATION_SAMPLES = 128
    ds_subset = ds.shuffle(seed=42).select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))
    print(f"Subset length: {len(ds_subset)}")

except Exception as e:
    print(f"Error: {e}")
