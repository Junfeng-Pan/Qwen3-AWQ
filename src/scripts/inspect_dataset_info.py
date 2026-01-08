from datasets import get_dataset_config_names

dataset_id = "neuralmagic/calibration"
try:
    configs = get_dataset_config_names(dataset_id)
    print(f"Available configs for {dataset_id}: {configs}")
except Exception as e:
    print(f"Error fetching configs: {e}")
