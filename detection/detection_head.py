import torch
import torch.nn as nn
import math

class DepthwiseSeparableConv(nn.Module):
    """
        Depth-wise separable convolution.
        A block of DepthwiseConv + PointwiseConv + BatchNorm + Activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=(kernel_size - 1) // 2, 
            groups=in_channels, 
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=False
        )

        # BatchNorm and Activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class NanoDetectionHead(nn.Module):
    """
        It takes multi-scale features and outputs a single concatenated tensor
        of predictions.
    """
    def __init__(self, num_classes, in_channels_list, head_width=32):
        super(NanoDetectionHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        self.head_width = head_width
        
        # Strides for each feature level
        self.strides = [8, 16, 32]
        
        # --- Stem Layers ---
        # 1x1 DepthwiseSeparableConv (pointwise) to unify channels from FPN levels
        self.stems = nn.ModuleList([
            DepthwiseSeparableConv(in_c, self.head_width, kernel_size=1, stride=1)
            for in_c in self.in_channels_list
        ])
        
        # --- Decoupled Branches (Shared Weights) ---
        
        # Classification Branch
        self.cls_convs = nn.Sequential(
            DepthwiseSeparableConv(self.head_width, self.head_width, kernel_size=3, stride=1),
            DepthwiseSeparableConv(self.head_width, self.head_width, kernel_size=3, stride=1),
        )
        # Final 1x1 conv for class predictions
        self.cls_pred = nn.Conv2d(
            self.head_width, self.num_classes, kernel_size=1, stride=1, padding=0
        )
        
        # Regression Branch (for BBox + Objectness)
        self.reg_convs = nn.Sequential(
            DepthwiseSeparableConv(self.head_width, self.head_width, kernel_size=3, stride=1),
            DepthwiseSeparableConv(self.head_width, self.head_width, kernel_size=3, stride=1),
        )
        # Final 1x1 conv for bbox regression
        self.reg_pred = nn.Conv2d(
            self.head_width, 4, kernel_size=1, stride=1, padding=0
        )
        # Final 1x1 conv for objectness
        self.obj_pred = nn.Conv2d(
            self.head_width, 1, kernel_size=1, stride=1, padding=0
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize the bias of the final classification/objectness layers
        # This helps training stability by starting with low confidence
        for m in [self.cls_pred, self.obj_pred]:
            if m.bias is not None:
                # Prior for "no object"
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(m.bias, bias_value)

    def forward(self, features):
        """
        Args:
            features (list[torch.Tensor]): List of 3 feature maps
                                           from the backbone.
        
        Returns:
            torch.Tensor: A single tensor of all predictions concatenated.
            Shape: [Batch, TotalGridCells, 5 + NumClasses]
                   (TotalGridCells = H/8*W/8 + H/16*W/16 + H/32*W/32)
                   The (5 + NumClasses) are [x, y, w, h, obj_conf, ...class_confs]
        """
        all_predictions = []
        
        # Iterate over each feature level (P3, P4, P5)
        for i, x in enumerate(features):
            # 1. Unify channels with the stem
            x_stem = self.stems[i](x)
            
            # 2. Pass through decoupled branches
            cls_feat = self.cls_convs(x_stem)
            reg_feat = self.reg_convs(x_stem)
            
            # 3. Get predictions
            cls_out = self.cls_pred(cls_feat)  # [B, NumClasses, H, W]
            reg_out = self.reg_pred(reg_feat)  # [B, 4, H, W]
            obj_out = self.obj_pred(reg_feat)  # [B, 1, H, W]
            
            # Combine all predictions for this level
            # Order: [reg (4), obj (1), cls (NumClasses)]
            level_predictions = torch.cat([reg_out, obj_out, cls_out], dim=1) # [B, H*W, 5 + NumClasses]
            all_predictions.append(level_predictions)
            
        # Return the list of predictions for all levels
        return all_predictions
