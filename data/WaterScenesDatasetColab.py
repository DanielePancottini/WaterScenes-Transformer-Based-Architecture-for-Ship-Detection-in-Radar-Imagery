import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import zipfile
import io

class WaterScenesDatasetFromZip(Dataset):
    """
    PyTorch Dataset che legge direttamente dagli ZIP:
    - image.zip
    - radar_revp_npy.zip
    - detection.zip
    """
    def __init__(self, zip_dir, split_file, image_transform=None, target_transform=None):
        """
        Args:
            zip_dir (str): cartella con i file ZIP
            split_file (str): file .txt con IDs dei campioni
        """
        self.zip_dir = zip_dir
        self.image_transform = image_transform
        self.target_transform = target_transform

        # Apri ZIP
        self.zips = {}
        for name in ["image.zip", "radar_revp_npy.zip", "detection.zip"]:
            path = Path(zip_dir) / name
            if not path.exists():
                raise FileNotFoundError(f"{path} non trovato")
            self.zips[name] = zipfile.ZipFile(path, 'r')

        # Lista file da split
        with open(split_file, 'r') as f:
            self.file_ids = [Path(line.strip()).stem for line in f if line.strip()]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        # --- 1. Carica immagine ---
        img_bytes = self.zips["image.zip"].read(f"{file_id}.jpg")
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # --- 2. Carica radar .npy ---
        radar_bytes = self.zips["radar_revp_npy.zip"].read(f"{file_id}.npy")
        radar_array = np.load(io.BytesIO(radar_bytes))
        radar_tensor = torch.tensor(radar_array, dtype=torch.float32)

        # --- 3. Carica label ---
        try:
            label_bytes = self.zips["detection.zip"].read(f"{file_id}.txt")
            labels = []
            for line in label_bytes.decode('utf-8').splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])
            label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0,5))
        except KeyError:
            label_tensor = torch.empty((0,5))

        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return {"image": image, "radar": radar_tensor, "label": label_tensor}
