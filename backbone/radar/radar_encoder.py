import math
import torch
import torch.nn as nn

from backbone.radar.deformable_conv import DeformableConv2d

image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class RadarConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RadarConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #Average pooling layer
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        #Deformable convolution layer
        self.deform_conv = DeformableConv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding= 3 // 2
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.deform_conv(x)
        return x

class RCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(RCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.radar_conv = RadarConv(in_channels, in_channels, stride=1)
        self.weight_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)

        if not downsample:
            self.weight_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.weight_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_residual = x
        x = self.radar_conv(x)
        x = self.weight_conv1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x_residual + x
        x = self.weight_conv2(x)
        return x

class RCNet(nn.Module):
    def __init__(self, in_channels, phi='S0'):
        super(RCNet, self).__init__()
        self.phi = phi
        self.in_channels = in_channels
       
        stage_blocks = []
        for i in range(4):
            if i == 0:
                stage_blocks.append(
                    RCBlock(
                        in_channels,
                        image_encoder_width[self.phi][i] // 4,
                        downsample=True,
                    )
                )
                stage_blocks.append(
                    RCBlock(
                        image_encoder_width[self.phi][i] // 4,
                        image_encoder_width[self.phi][i] // 4,
                        downsample=True,
                    )
                )
            else:
                stage_blocks.append(
                    RCBlock(
                        image_encoder_width[self.phi][i - 1] // 4,
                        image_encoder_width[self.phi][i - 1] // 4,
                        downsample=False,
                    )
                )
                stage_blocks.append(
                    RCBlock(
                        image_encoder_width[self.phi][i - 1] // 4,
                        image_encoder_width[self.phi][i] // 4,
                        downsample=True,
                    )
                )
        
        self.stage_blocks = nn.Sequential(*stage_blocks)

    def forward_stages(self, x):
        outputs = []
        for i in range(len(self.stage_blocks)):
            x = self.stage_blocks[i](x)
            if i > 1 and i % 2 == 1:
                outputs.append(x)
        return outputs

    def forward(self, x):
        x = self.forward_stages(x)
        return x
    
class RCNetWithTransformer(nn.Module):
    def __init__(self, in_channels, phi='S0', num_transformer_layers=2, num_heads=4, max_input_hw=680):
        super(RCNetWithTransformer, self).__init__()

        #initialize RCNet
        self.rcnet = RCNet(in_channels, phi)

        #get the output channels from RCNet
        width = image_encoder_width[phi]
        self.channels = [
            width[1] // 4,  # C3 (stride 8)
            width[2] // 4,  # C4 (stride 16)
            width[3] // 4,  # C5 (stride 32)
        ]

        #Positional embeddings for transformer
        self.positional_embedding = nn.ParameterList()
        strides = [8, 16, 32]

        #Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList()

        # Create positional embeddings and transformer blocks for each scale
        for i, d_model in enumerate(self.channels):
            pe = nn.Parameter(torch.randn(1, d_model, math.ceil(max_input_hw / strides[i]), math.ceil(max_input_hw / strides[i])))
            self.positional_embedding.append(pe)

            transformer_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=self.channels[i] * 4,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )

            transformer_encoder = nn.TransformerEncoder(
                transformer_layer,
                num_layers=num_transformer_layers
            )

            self.transformer_blocks.append(transformer_encoder)

    def forward(self, x):

        # Pass input through RCNet to get multi-scale features
        features = self.rcnet(x)

        # Process each scale with its corresponding transformer block
        transformed_features = []

        for i, feature in enumerate(features):
            B, C, H, W = feature.shape

            # Add positional embedding
            pe_cropped = self.positional_embedding[i][:, :, :H, :W]
            feature = feature + pe_cropped

            # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
            feature_reshaped = feature.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # Pass through transformer block
            transformed_feature = self.transformer_blocks[i](feature_reshaped)

            # Reshape back to (B, C, H, W)
            transformed_feature = transformed_feature.reshape(B, H, W, C).permute(0, 3, 1, 2)

            # Add residual connection
            transformed_feature = transformed_feature + feature

            transformed_features.append(transformed_feature)

        return transformed_features