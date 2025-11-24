import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision

# --- Imports from your project ---
from backbone.radar.radar_encoder import RCNetWithTransformer
from detection.detection_head import NanoDetectionHead
from model import RadarDetectionModel
from data.WaterScenesDataset import WaterScenesDataset

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/rcnet_radar_detection_half_transformer.pth"
DATASET_ROOT = os.path.abspath("./data/WaterScenes")
TEST_FILE = os.path.join(DATASET_ROOT, "test.txt")

# Stats (Must match training)
RADAR_MEAN = [0.1127, -0.0019, -0.0012, 0.0272]
RADAR_STD  = [3.1396,  0.2177,  0.0556,  0.6252]
NUM_CLASSES = 7
CONF_THRESH = 0.25  # Only show boxes with > 25% confidence
NMS_THRESH = 0.45   # IoU threshold for removing duplicate boxes

def decode_outputs(outputs, input_shape=(320, 320)):
    """
    Decodes the raw output of the model into [x1, y1, x2, y2, score, class]
    """
    grids = []
    strides = [8, 16, 32]
    decoded_boxes = []

    for i, output in enumerate(outputs):
        # output shape: [Batch, C, H, W]
        B, C, H, W = output.shape
        stride = strides[i]
        
        # Create Grid
        yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, -1, 2).to(output.device).float()
        
        # Flatten output: [B, H*W, C]
        output = output.flatten(start_dim=2).permute(0, 2, 1)
        
        # Decode XY
        # (x_offset + grid_x) * stride
        output[..., :2] = (output[..., :2] + grid) * stride
        
        # Decode WH
        # exp(w_offset) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        
        decoded_boxes.append(output)

    # Concat all layers: [B, Total_Anchors, 5 + Num_Classes]
    all_preds = torch.cat(decoded_boxes, dim=1)
    return all_preds

def non_max_suppression(prediction, conf_thres=0.25, nms_thres=0.45):
    """
    Applies NMS to filter boxes.
    prediction: [Total_Anchors, 5 + Num_Classes]
    """
    # 1. Filter by Objectness Confidence
    # obj_conf * cls_conf
    box_corner = prediction.new(prediction.shape)
    
    # Convert cx, cy, w, h -> x1, y1, x2, y2
    box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
    box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
    box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
    box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
    prediction[:, :4] = box_corner[:, :4]

    output = [None]
    
    # Get Score = Obj_Conf * Sigmoid(Best_Class_Score)
    # Note: Your Head might calculate obj/cls differently. 
    # Assuming the output is [x, y, w, h, obj, class1, class2...]
    
    # Apply sigmoid to obj and classes
    obj_score = torch.sigmoid(prediction[:, 4])
    cls_score = torch.sigmoid(prediction[:, 5:])
    
    # Find max class
    max_cls_score, max_cls_idx = torch.max(cls_score, 1)
    
    # Final score
    final_score = obj_score * max_cls_score
    
    # Filter
    mask = final_score > conf_thres
    pred_filtered = prediction[mask]
    scores_filtered = final_score[mask]
    class_filtered = max_cls_idx[mask]
    
    if pred_filtered.size(0) == 0:
        return []

    # Prepare for NMS: [x1, y1, x2, y2]
    boxes = pred_filtered[:, :4]
    
    # NMS
    keep = torchvision.ops.nms(boxes, scores_filtered, nms_thres)
    
    # Result: [x1, y1, x2, y2, score, class]
    results = []
    for idx in keep:
        res = torch.cat((boxes[idx], scores_filtered[idx].unsqueeze(0), class_filtered[idx].float().unsqueeze(0)))
        results.append(res)
        
    return torch.stack(results)

def main():
    # 1. Load Dataset (Validation)
    image_transform = transforms.Compose([
        transforms.Resize((320, 320)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = WaterScenesDataset(
        root_dir=DATASET_ROOT,
        split_file=TEST_FILE,
        image_transform=image_transform,
        radar_mean=RADAR_MEAN,
        radar_std=RADAR_STD
    )

    # 2. Load Model
    # IMPORTANT: Ensure this matches the architecture you trained with!
    # If you used RCNetWithTransformer in main.py, keep it here.
    backbone = RCNetWithTransformer(in_channels=4, phi='S0', max_input_hw=320)
    head = NanoDetectionHead(num_classes=NUM_CLASSES, in_channels_list=[12, 24, 44], head_width=32)
    model = RadarDetectionModel(backbone, head).to(DEVICE)
    
    print(f"Loading weights from {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 3. Inference Loop (Show 3 random examples)
    indices = np.random.choice(len(dataset), 3, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        file_id = sample['file_id']
        radar = sample['radar'].unsqueeze(0).to(DEVICE) # Add batch dim
        
        # Run Model
        with torch.no_grad():
            preds = model(radar) # List of 3 tensors
            decoded = decode_outputs(preds) # [1, 2100, 5+Nc]
            results = non_max_suppression(decoded[0], CONF_THRESH, NMS_THRESH)

        # Load real image for plotting
        img_path = os.path.join(DATASET_ROOT, 'image', f"{file_id}.jpg")
        orig_img = Image.open(img_path).convert("RGB").resize((320, 320))
        img_draw = np.array(orig_img)

        # Draw Boxes
        if len(results) > 0:
            for box in results:
                x1, y1, x2, y2, score, cls_id = box.cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw Rectangle
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label
                label = f"Cls {int(cls_id)}: {score:.2f}"
                cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"No objects detected for {file_id} (Try lowering CONF_THRESH)")

        # Plot Radar + Result
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show Radar (Power Channel)
        radar_power = sample['radar'][3].cpu().numpy()
        ax[0].imshow(radar_power, cmap='viridis')
        ax[0].set_title("Radar Power Channel")
        
        # Show Prediction
        ax[1].imshow(img_draw)
        ax[1].set_title(f"Prediction (ID: {file_id})")
        
        plt.show()

if __name__ == "__main__":
    main()