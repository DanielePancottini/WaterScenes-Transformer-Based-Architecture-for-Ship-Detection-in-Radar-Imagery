import os
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from backbone.radar.radar_encoder import RCNet, RCNetWithTransformer
from data.WaterScenesDataset import WaterScenesDataset
from data.WaterScenesDataset import collate_fn
from preprocess.revp import REVP_Transform

# --- Set your paths ---
DATASET_ROOT = "./data/WaterScenes"
TRAIN_FILE = os.path.join(DATASET_ROOT, "train.txt")
VAL_FILE = os.path.join(DATASET_ROOT, "val.txt")
TEST_FILE = os.path.join(DATASET_ROOT, "test.txt")

# --- Define transforms ---
# This is your final model input size
TARGET_SIZE = (680, 680) 

# --- Image Transforms ---
# (Unchanged from before)
image_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Radar Transform ---
# This is now just one clean step.
# It handles creation AND resizing.
radar_transform = REVP_Transform(target_size=TARGET_SIZE)

# --- 1. Create the Datasets ---
train_dataset = WaterScenesDataset(
    root_dir=DATASET_ROOT,
    split_file=TRAIN_FILE,
    image_transform=image_transform,
    radar_transform=radar_transform,
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

# Parameters
in_channels = 4
out_channels = 8
stride = 1
height = 680
width = 680

rcnet = RCNet(in_channels)

rcnet_tf = RCNetWithTransformer(
    in_channels=in_channels, 
    phi='S0',
    num_transformer_layers=2,
    num_heads=4,
    max_input_hw=680 # Set max_input_hw to 256 for this example
)


# (Set up val_loader similarly)

#I want to test my dataloader
if __name__ == "__main__":
    
    print("Testing the DataLoader...")
    
    # Function to un-normalize for plotting
    def unnormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    # --- Loop over batches ---
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"--- Batch {batch_idx + 1} ---")

        # --- 1. Get Batched Tensors ---
        images_batch = batch_data['image']    # Shape [B, 3, H, W]
        radars_batch = batch_data['radar']    # Shape [B, 4, H, W]
        labels_batch = batch_data['label']

        print(f"Image batch shape: {images_batch.shape}")
        print(f"Radar batch shape: {radars_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")

        # --- 2. Select the FIRST item from the batch for plotting ---
        image_tensor = images_batch[0]  # Shape [3, H, W]
        radar_tensor = radars_batch[0]  # Shape [4, H, W]

        # --- 3. Prepare for Plotting ---
        # Un-normalize the image tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_to_plot = unnormalize(image_tensor.clone(), mean, std) # Use .clone()
        image_to_plot = image_to_plot.permute(1, 2, 0).numpy()
        
        # Clip values to [0, 1] for valid imshow
        image_to_plot = np.clip(image_to_plot, 0, 1)

        # Split the 4 radar channels
        range_ch = radar_tensor[0].numpy()
        elevation_ch = radar_tensor[1].numpy()
        doppler_ch = radar_tensor[2].numpy()
        power_ch = radar_tensor[3].numpy()

        # Create a list for plotting
        titles = ['Original Image (Un-normalized)', 'REVP: Range', 'REVP: Elevation', 'REVP: Doppler', 'REVP: Power']
        images = [image_to_plot, range_ch, elevation_ch, doppler_ch, power_ch]

        # --- 4. Plot the results ---
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        for i, (img, title) in enumerate(zip(images, titles)):
            ax = axes[i]
            if i == 0:
                ax.imshow(img) # Show the RGB image
            else:
                # Show the radar channels
                if title == 'REVP: Range':
                    im = ax.imshow(img, cmap='viridis', vmax=100) # Cap at 100m
                else:
                    im = ax.imshow(img, cmap='viridis')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(title)
            ax.axis('off')

        plt.suptitle(f"Batch {batch_idx + 1}, Item 0")
        plt.tight_layout()
        plt.show()

        print("--- Testing original RCNet ---")
        features_original = rcnet(radars_batch)
        print("Original RCNet output shapes:")
        for f in features_original:
            print(f.shape)
        
        print("\n--- Testing RCNetWithTransformer ---")
        features_transformed = rcnet_tf(radars_batch)
        print("Transformed RCNet output shapes:")
        for f in features_transformed:
            print(f.shape)

        # 5. Verify shapes are the same
        for f_orig, f_tf in zip(features_original, features_transformed):
            assert f_orig.shape == f_tf.shape
        
        print("\nSuccess! Output feature maps have the same shape.")
                
        if batch_idx == 0:  # Plot first 2 batches
            print("--- Test complete ---")
            break