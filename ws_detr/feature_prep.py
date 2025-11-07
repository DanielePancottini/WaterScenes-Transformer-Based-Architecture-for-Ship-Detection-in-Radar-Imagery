"""
Utilities for preparing image and REVP features for point-based fusion.

Provides:
- image_to_pointset: convert RGB image (3,H,W) -> points (5, N) (rgb + x,y)
- revp_to_pointset: convert REVP map (4,H,W) -> points (6, N) (radar features + x,y)
- ECALayer: Efficient Channel Attention (ECA) module
- SpatialAttention (Deformable if available, else conv-based fallback)
- apply_revp_attention: convenience function applying channel+spatial attention to a REVP map

Position encoding matches the paper: for pixel at (i,j) with image width W and height H,
we use x = i / W - 0.5, y = j / H - 0.5. Note: this module assumes pixel indices start at 0.

Example:
    import torch
    from ws_detr.feature_prep import image_to_pointset, revp_to_pointset, apply_revp_attention

    img = torch.rand(3, 720, 1280)          # C,H,W
    revp = torch.rand(4, 720, 1280)         # C,H,W

    img_pts = image_to_pointset(img)        # (5, H*W)
    revp_att = apply_revp_attention(revp)   # (4, H, W) attention-applied
    revp_pts = revp_to_pointset(revp_att)   # (6, H*W)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _coords_hw(H: int, W: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return positional coordinates for pixels as tensor shape (2, H, W): (x, y).

    x corresponds to i/W - 0.5 where i in [0..W-1]; y corresponds to j/H - 0.5 where j in [0..H-1].
    """
    # grid_x: shape (W,) values 0..W-1
    xs = (torch.arange(W, device=device, dtype=dtype) / float(W)) - 0.5
    ys = (torch.arange(H, device=device, dtype=dtype) / float(H)) - 0.5
    grid_x = xs.unsqueeze(0).expand(H, W)  # (H, W)
    grid_y = ys.unsqueeze(1).expand(H, W)  # (H, W)
    coords = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
    return coords


def image_to_pointset(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image tensor (3, H, W) into a point set of shape (5, N) where N = H*W.
    Each point contains [r, g, b, x_pos, y_pos].

    Accepts numpy arrays (will be converted to torch tensors).
    Preserves device and dtype where possible.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor or numpy.ndarray")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape (3, H, W)")

    device = image.device
    dtype = image.dtype if image.dtype.is_floating_point else torch.float32

    C, H, W = image.shape
    coords = _coords_hw(H, W, device=device, dtype=dtype)  # (2, H, W)

    # Flatten
    img_flat = image.reshape(3, -1)  # (3, N)
    coords_flat = coords.reshape(2, -1)  # (2, N)

    pts = torch.cat([img_flat.to(dtype=dtype, device=device), coords_flat.to(dtype=dtype, device=device)], dim=0)
    # shape (5, N)
    return pts


def revp_to_pointset(revp: torch.Tensor) -> torch.Tensor:
    """
    Convert a REVP tensor (4, H, W) into a point set of shape (6, N) where N = H*W.
    Each point contains [r_range, elevation, velocity, power, x_pos, y_pos].

    Accepts numpy arrays (will be converted to torch tensors).
    Preserves device and dtype where possible.
    """
    if isinstance(revp, np.ndarray):
        revp = torch.from_numpy(revp)
    if not torch.is_tensor(revp):
        raise TypeError("revp must be a torch.Tensor or numpy.ndarray")
    if revp.ndim != 3 or revp.shape[0] != 4:
        raise ValueError("revp must have shape (4, H, W)")

    device = revp.device
    dtype = revp.dtype if revp.dtype.is_floating_point else torch.float32

    C, H, W = revp.shape
    coords = _coords_hw(H, W, device=device, dtype=dtype)  # (2, H, W)

    revp_flat = revp.reshape(4, -1)
    coords_flat = coords.reshape(2, -1)

    pts = torch.cat([revp_flat.to(dtype=dtype, device=device), coords_flat.to(dtype=dtype, device=device)], dim=0)
    # shape (6, N)
    return pts


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) as in "ECA-Net" (Wang et al., 2020).

    This implementation follows the simple idea: global avg pool -> 1D conv -> sigmoid -> scale channels.
    The kernel size k can be tuned; paper suggests using an adaptively determined k but we expose a small int.
    """

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.channels = channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (C, H, W) -> handle both
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        B, C, H, W = x.shape
        y = self.avgpool(x)  # (B, C, 1, 1)
        y = y.view(B, 1, C)  # (B, 1, C)
        y = self.conv(y)     # (B, 1, C)
        y = self.sigmoid(y).view(B, C, 1, 1)
        out = x * y
        if squeeze:
            out = out.squeeze(0)
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module. Attempts to use DeformConv2d if torchvision provides it; otherwise falls
    back to a conv-based attention map.

    If DeformConv2d is available, we predict offsets via a small conv and apply deformable conv then
    multiply the result as an attention map. Otherwise, we predict a single-channel attention map
    via conv -> sigmoid and multiply.
    """

    def __init__(self, in_channels: int, mid_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.mid = mid_channels

        # Try to import DeformConv2d
        try:
            from torchvision.ops import DeformConv2d
            self._deformable_available = True
            # offset conv predicts 2*k*k offsets; for kernel=3 we need 18 offsets
            self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
            # deform conv will map in_channels -> in_channels (preserve channels)
            self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
            # small conv to reduce to attention map
            self.reduce_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        except Exception:
            self._deformable_available = False
            # fallback
            self.att_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.mid, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid, 1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) or (C, H, W)
        returns same shape as input (attention-applied)
        """
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        B, C, H, W = x.shape

        if self._deformable_available:
            offsets = self.offset_conv(x)
            # deform conv
            d = self.deform_conv(x, offsets)
            att = torch.sigmoid(self.reduce_conv(d))  # (B,1,H,W)
            out = x * att
        else:
            att = self.att_conv(x)  # (B,1,H,W)
            out = x * att

        if squeeze:
            out = out.squeeze(0)
        return out


def apply_revp_attention(revp: torch.Tensor, k_size: int = 3) -> torch.Tensor:
    """
    Apply channel (ECA) and spatial attention to a REVP tensor (4, H, W) or (B, 4, H, W).
    Returns the attention-applied REVP tensor with same shape.
    """
    single = False
    if isinstance(revp, np.ndarray):
        revp = torch.from_numpy(revp)
    if not torch.is_tensor(revp):
        raise TypeError("revp must be a torch.Tensor or numpy.ndarray")

    if revp.ndim == 3:
        revp = revp.unsqueeze(0)
        single = True
    if revp.ndim != 4:
        raise ValueError("revp must be shape (4,H,W) or (B,4,H,W)")

    B, C, H, W = revp.shape
    eca = ECALayer(C, k_size=k_size).to(revp.device)
    sa = SpatialAttention(C).to(revp.device)

    out = eca(revp)
    out = sa(out)

    if single:
        out = out.squeeze(0)
    return out
