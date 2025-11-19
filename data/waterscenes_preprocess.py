import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# --- IMPORTANT ---
# Import your REVP_Transform class
# Adjust the path if 'preprocess.revp' is incorrect.
from preprocess.revp import REVP_Transform

# --- Configuration ---
DATASET_ROOT = os.path.abspath("./data/WaterScenes") # Your main dataset folder
TARGET_SIZE = (320, 320)           # Must match your training config

# Define input directories
IMAGE_DIR = os.path.join(DATASET_ROOT, 'image')
RADAR_DIR = os.path.join(DATASET_ROOT, 'radar')
SPLIT_FILES = [
    os.path.join(DATASET_ROOT, "train.txt"),
    os.path.join(DATASET_ROOT, "val.txt"),
    os.path.join(DATASET_ROOT, "test.txt")
]

# Define NEW output directory
RADAR_REVP_DIR = os.path.join(DATASET_ROOT, "radar_revp_npy")

ORIGINAL_IMAGE_SIZE = (1080, 1920)  # H, W tuple
# ---------------------

def _load_file_ids(split_file_path):
    """Helper function to read a split file and return a list of IDs."""
    if not os.path.exists(split_file_path):
        print(f"Warning: Split file not found {split_file_path}")
        return []
    
    with open(split_file_path, 'r') as f:
        file_ids = []
        for line in f:
            line = line.strip()
            if line:
                file_id = Path(line).stem
                file_ids.append(file_id)
    return file_ids

def main():
    """
    Main preprocessing function.
    Loads raw CSV radar data, applies REVP_Transform,
    and saves the resulting tensor as a .npy file.
    """
    
    # 1. Create the output directory if it doesn't exist
    os.makedirs(RADAR_REVP_DIR, exist_ok=True)
    print(f"Output directory created at: {RADAR_REVP_DIR}")
    
    # 2. Initialize the REVP Transform
    # This is the same transform you used in main.py
    radar_transform = REVP_Transform(target_size=TARGET_SIZE)

    # 3. Load all unique file IDs from train, val, and test splits
    all_file_ids = set()
    for f in SPLIT_FILES:
        ids = _load_file_ids(f)
        all_file_ids.update(ids)
    
    print(f"Found {len(all_file_ids)} unique samples to process.")
    
    # 4. Loop over all files and process them
    for file_id in tqdm(all_file_ids):
        
        # --- Define paths ---
        csv_path = os.path.join(RADAR_DIR, f"{file_id}.csv")
        output_npy_path = os.path.join(RADAR_REVP_DIR, f"{file_id}.npy")

        # --- Check if already processed ---
        if os.path.exists(output_npy_path):
            continue

        try:
            # --- b) Load raw radar CSV (The slow part) ---
            try:
                radar_df = pd.read_csv(csv_path)
                radar_points = radar_df[['u', 'v', 'range', 'elevation', 'doppler', 'power']].values
            except Exception:
                # Handle empty/corrupt CSVs
                radar_points = np.empty((0, 6), dtype=np.float32)
            
            radar_tensor_raw = torch.tensor(radar_points, dtype=torch.float32)

            # --- c) Apply the REVP transform (The CPU-intensive part) ---
            # This converts [N, 6] points -> [4, H, W] tensor
            radar_revp_tensor = radar_transform(radar_tensor_raw, ORIGINAL_IMAGE_SIZE)

            # --- d) Save the FINAL tensor as a .npy file ---
            # Convert to numpy for saving
            radar_revp_numpy = radar_revp_tensor.numpy()
            np.save(output_npy_path, radar_revp_numpy)

        except Exception as e:
            print(f"\nError processing {file_id}: {e}")
            print(f"  CSV path: {csv_path}")
            
    print("\n--- Preprocessing complete! ---")
    print(f"All processed REVP maps are saved in: {RADAR_REVP_DIR}")

if __name__ == "__main__":
    main()