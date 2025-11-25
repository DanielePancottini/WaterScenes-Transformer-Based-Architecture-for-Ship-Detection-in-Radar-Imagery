import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from backbone.radar.radar_encoder import RCNet, RCNetWithTransformer
from data.WaterScenesDataset import WaterScenesDataset
from data.WaterScenesDataset import collate_fn
from detection.detection_head import NanoDetectionHead
from model import RadarDetectionModel
from trainer import Trainer
import torch.optim as optim
from detection.detection_loss import YOLOLoss, ModelEMA, get_lr_scheduler
from PIL import Image

# --- Set your paths ---
DATASET_ROOT = os.path.abspath("./data/WaterScenes")
TRAIN_FILE = os.path.join(DATASET_ROOT, "train.txt")
VAL_FILE = os.path.join(DATASET_ROOT, "val.txt")
TEST_FILE = os.path.join(DATASET_ROOT, "test.txt")
MODEL_SAVE_PATH = os.path.abspath("./checkpoints/rcnet_radar_detection_half_transformer_transfer_learning_30e.pth")

# --- Config ---
TARGET_SIZE = (320, 320) 
NUM_CLASSES = 7 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 32
STRIDES = [8, 16, 32] 
IN_CHANNELS = 4
IN_CHANNELS_LIST = [12, 24, 44]
HEAD_WIDTH = 32
INITIAL_LR = 0.03
MOMENTUM = 0.937
FP16 = False
RADAR_MEAN = [0.1127, -0.0019, -0.0012, 0.0272]
RADAR_STD  = [3.1396,  0.2177,  0.0556,  0.6252]

# --- Add CuDNN Benchmark ---
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    print(f"Using Device: {DEVICE}")
    
    # --- Image Transforms ---
    image_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Create the Datasets ---
    print("Initializing Datasets...")

    train_dataset = WaterScenesDataset(
        root_dir=DATASET_ROOT,
        split_file=TRAIN_FILE,
        image_transform=image_transform,
        radar_mean=RADAR_MEAN,
        radar_std=RADAR_STD
    )

    validation_dataset = WaterScenesDataset(
        root_dir=DATASET_ROOT,
        split_file=VAL_FILE,
        image_transform=image_transform,
        radar_mean=RADAR_MEAN,
        radar_std=RADAR_STD
    )

    # --- Create the DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # ==========================================
    #      VISUALIZE FIRST EXAMPLE ONLY
    # ==========================================
    print("\n--- Visualizing First Dataset Example ---")
    
    # Fetch a single batch from the loader
    batch_data = next(iter(train_loader))
    
    radars_batch = batch_data['radar']
    labels_batch = batch_data['label']
    
    file_ids = batch_data['file_ids']
    current_id = file_ids[0]
    
    # Load the RGB Image Manually from Disk
    img_path = os.path.join(DATASET_ROOT, 'image', f"{current_id}.jpg")
    try:
        real_image = Image.open(img_path).convert('RGB')
        real_image = real_image.resize(TARGET_SIZE)
        image_to_plot = np.array(real_image)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        image_to_plot = np.zeros((320, 320, 3)) # Black image fallback

    # Get Radar Channels for the first item
    radar_tensor = radars_batch[0] # [4, H, W]
    
    # Plot
    titles = [f'RGB ({current_id})', 'Range', 'Elevation', 'Doppler', 'Power']
    images_list = [
        image_to_plot, 
        radar_tensor[0].numpy(), 
        radar_tensor[1].numpy(), 
        radar_tensor[2].numpy(), 
        radar_tensor[3].numpy()
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (img, title) in enumerate(zip(images_list, titles)):
        ax = axes[i]
        if i == 0:
            ax.imshow(img) # RGB
        else:
            im = ax.imshow(img, cmap='viridis') # Radar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Visualization closed. Proceeding to Model Setup...")

    # ==========================================
    #           MODEL SETUP & TRAIN
    # ==========================================

    # Initialize RCNet with Transformer Backbone
    rcnet_tf = RCNetWithTransformer(
        in_channels=IN_CHANNELS, 
        phi='S0',
        num_transformer_layers=2,
        num_heads=4,
        max_input_hw=320
    )

    # --- Initialize Head ---
    head = NanoDetectionHead(
        num_classes=NUM_CLASSES,
        in_channels_list=IN_CHANNELS_LIST,
        head_width=HEAD_WIDTH
    )

    # --- Initialize Model ---
    model = RadarDetectionModel(backbone=rcnet_tf, detection_head=head)
    model.to(DEVICE)

    # --- Load Weights ---
    # LOAD WEIGHTS FROM PREVIOUS RUN
    checkpoint = torch.load("./checkpoints/rcnet_radar_detection_half_transformer_20e.pth")
    model.load_state_dict(checkpoint)
    print("Loaded weights from previous training!")
        
    # EMA and Loss
    ema = ModelEMA(model)
    criterion = YOLOLoss(
        num_classes=NUM_CLASSES,
        strides=STRIDES,
        fp16=FP16 
    ).to(DEVICE)

    # --- Optimizer ---
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=5e-4)

    # --- Learning rate scaler ---
    nbs = 64
    lr_limit_max = 5e-2 
    lr_limit_min = 5e-4
    Init_lr_fit = min(max(BATCH_SIZE / nbs * INITIAL_LR, lr_limit_min), lr_limit_max)
    Min_lr_fit = Init_lr_fit * 0.01

    # --- Learning Rate Scheduler ---
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    lr_scheduler = get_lr_scheduler(
        lr_decay_type="cos",
        lr=Init_lr_fit,
        min_lr=Min_lr_fit,
        total_iters=total_steps,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE,
        ema=ema,
        lr_scheduler=lr_scheduler,
        fp16=FP16
    )
    
    print("Starting training...")
    trainer.train(final_model_path=MODEL_SAVE_PATH)
    print("Training complete.")