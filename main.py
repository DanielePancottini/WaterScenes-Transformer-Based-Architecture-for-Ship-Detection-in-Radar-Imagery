import os
from torch.utils.data import DataLoader
from torchvision import transforms
from WaterScenesDataset import WaterScenesDataset
from WaterScenesDataset import collate_fn

# --- Set your paths ---
DATASET_ROOT = "./data/WaterScenes"
TRAIN_FILE = os.path.join(DATASET_ROOT, "train.txt")
VAL_FILE = os.path.join(DATASET_ROOT, "val.txt")
TEST_FILE = os.path.join(DATASET_ROOT, "test.txt")

# --- Define transforms ---
image_preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # Example size for YOLO
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# --- 1. Create the Datasets ---
train_dataset = WaterScenesDataset(
    root_dir=DATASET_ROOT,
    split_file=TRAIN_FILE,
    image_transform=image_preprocess
    # You can add transforms for radar or labels here
)

# --- 2. Create the DataLoaders ---
BATCH_SIZE = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# (Set up val_loader similarly)

#I want to test my dataloader
if __name__ == "__main__":
    
    print("Testing the DataLoader...")
    
    # 'batch_data' will be the dictionary: {'image': ..., 'radar': ..., 'label': ...}
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"--- Batch {batch_idx + 1} ---")

        # Now, access the items using their keys
        images = batch_data['image']
        radars = batch_data['radar']  # This is a LIST of tensors
        labels = batch_data['label']  # This is ONE TENSOR [Total_Obj, 6]

        # --- Corrected Print Statements ---
        print(f"Images shape: {images.shape}")  # Expecting [B, C, H, W]
        
        # 'radars' is a LIST, so we print its length (batch size)
        # and the shape of the first tensor in the list
        print(f"Radars: {len(radars)} tensors in a list")
        if len(radars) > 0:
            print(f"  Shape of first radar tensor: {radars[0].shape}") # e.g., [N_points, 5]
        
        # 'labels' is a SINGLE TENSOR
        print(f"Labels shape: {labels.shape}") # e.g., [Total_Objects_In_Batch, 6]
        
        if batch_idx == 10:  # Just test first 3 batches
            print("--- Test complete ---")
            break