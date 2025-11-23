import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data.WaterScenesDataset import WaterScenesDataset, collate_fn

# --- Config ---
DATASET_ROOT = os.path.abspath("./data/WaterScenes")
TRAIN_FILE = os.path.join(DATASET_ROOT, "train.txt")
BATCH_SIZE = 16

def calculate_stats():
    print("Initializing Training Dataset...")
    # Note: We do NOT pass image_transform because we only care about Radar here
    train_dataset = WaterScenesDataset(
        root_dir=DATASET_ROOT,
        split_file=TRAIN_FILE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    print("Calculating mean and std (this may take a moment)...")
    
    channels_sum = torch.zeros(4)
    channels_sq_sum = torch.zeros(4)
    num_batches = 0
    total_pixels = 0

    for batch in tqdm(train_loader):
        # radar shape: [B, 4, H, W]
        radar = batch['radar']
        
        # Calculate sum and squared sum per channel (dim 0, 2, 3 are B, H, W)
        # Result shape: [4]
        channels_sum += torch.sum(radar, dim=[0, 2, 3])
        channels_sq_sum += torch.sum(radar ** 2, dim=[0, 2, 3])
        
        # Count total pixels per channel
        B, C, H, W = radar.shape
        total_pixels += B * H * W

    # Final calculations
    mean = channels_sum / total_pixels
    std = (channels_sq_sum / total_pixels - mean ** 2).sqrt()

    print("\n" + "="*30)
    print("RESULTS TO COPY INTO MAIN.PY")
    print("="*30)
    print(f"RADAR_MEAN = {mean.tolist()}")
    print(f"RADAR_STD  = {std.tolist()}")
    print("="*30)

if __name__ == "__main__":
    calculate_stats()