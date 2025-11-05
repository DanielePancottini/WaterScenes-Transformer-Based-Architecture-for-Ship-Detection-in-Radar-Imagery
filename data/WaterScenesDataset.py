import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path

class WaterScenesDataset(Dataset):
    """
    Custom PyTorch Dataset for the WaterScenes dataset.
    Loads hybrid data (Image + 4D Radar) and corresponding 
    YOLO-format detection labels.
    """
    def __init__(self, root_dir, split_file, image_transform=None, radar_transform=None, target_transform=None):
        """
        Args:
            root_dir (str): Path to the main dataset directory.
            split_file (str): Path to the .txt file (e.g., 'train.txt') 
                              containing the file IDs for this split.
            image_transform (callable, optional): Transform to be applied to the image.
            radar_transform (callable, optional): Transform to be applied to the radar data.
            target_transform (callable, optional): Transform to be applied to the label.
        """
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.radar_transform = radar_transform
        self.target_transform = target_transform
        
        # --- Define file paths ---
        self.image_dir = os.path.join(root_dir, 'image')
        self.radar_dir = os.path.join(root_dir, 'radar')
        self.label_dir = os.path.join(root_dir, 'detection', 'yolo')

        # Load the file IDs (e.g., '000001', '000002') from the split file
        self.file_ids = self._load_file_ids(split_file)

    def _load_file_ids(self, split_file_path):
        """Helper function to read the .txt file and return a list of IDs."""
        with open(split_file_path, 'r') as f:
            # Process each line
            file_ids = []
            for line in f:
                line = line.strip()
                if line:
                    file_id = Path(line).stem
                    file_ids.append(file_id)
        return file_ids

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_ids)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        A sample consists of an image, radar data, and a label.
        """
        # Get the file ID for this index
        file_id = self.file_ids[idx]
        
        # --- 1. Load Image ---
        img_path = os.path.join(self.image_dir, f"{file_id}.jpg")
        image = Image.open(img_path).convert('RGB')

        # --- 2. Load 4D Radar Data ---
        radar_path = os.path.join(self.radar_dir, f"{file_id}.csv")
        try:
            radar_df = pd.read_csv(radar_path)
            radar_points = radar_df[['x', 'y', 'z', 'doppler', 'power']].values
        except Exception as e:
            # print(f"Error loading radar file {radar_path}: {e}")
            radar_points = np.empty((0, 5), dtype=np.float32)
            
        radar_tensor = torch.tensor(radar_points, dtype=torch.float32)

        # --- 3. Load Label (YOLO format) ---
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        labels = []
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # [class_id, x_c, y_c, w, h]
                            labels.append([float(p) for p in parts])
            except Exception as e:
                print(f"Error loading label file {label_path}: {e}")
        
        # Convert to a tensor
        # If no objects, create an empty tensor of the correct shape [0, 5]
        if not labels:
            label_tensor = torch.empty((0, 5), dtype=torch.float32)
        else:
            label_tensor = torch.tensor(labels, dtype=torch.float32)

        # --- 4. Apply Transforms ---
        if self.image_transform:
            image = self.image_transform(image)
        if self.radar_transform:
            radar_tensor = self.radar_transform(radar_tensor)
        if self.target_transform:
            # Note: target_transform now operates on a [N_objects, 5] tensor
            label_tensor = self.target_transform(label_tensor)

        # Return data as a dictionary
        sample = {
            'image': image,
            'radar': radar_tensor,
            'label': label_tensor # Shape: [N_objects, 5]
        }
        
        return sample
    
def collate_fn(batch):
    """
    Custom collate function to handle variable-length radar data
    AND variable-length YOLO labels.
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Radar data is a list of tensors (one for each item in the batch)
    radar_data = [item['radar'] for item in batch]
    
    # Process labels (YOLO format)
    labels = []
    for i, item in enumerate(batch):
        label = item['label']  # This is [N_objects, 5]
        
        # Check if there are any objects in this sample
        if label.shape[0] > 0:
            # Create a tensor for the batch index, shape [N_objects, 1]
            batch_idx_col = torch.full((label.shape[0], 1), i)
            
            # Prepend batch index: [batch_idx, class_id, x, y, w, h]
            label_with_idx = torch.cat((batch_idx_col, label), dim=1)
            labels.append(label_with_idx)
    
    # Concatenate all labels from all batch items into one tensor
    if labels:
        labels_tensor = torch.cat(labels, 0)
    else:
        # No objects in this entire batch
        labels_tensor = torch.empty((0, 6), dtype=torch.float32)
    
    return {
        'image': images,
        'radar': radar_data,
        'label': labels_tensor  # Shape: [Total_Objects_In_Batch, 6]
    }