import torch
import torch.nn as nn
import numpy as np

class REVP_Transform(nn.Module):
    def __init__(self, target_size=(680, 680)):
        super().__init__()
        self.target_size = target_size
        self.H_target, self.W_target = int(target_size[0]), int(target_size[1])

    def forward(self, radar_points, original_image_size):
        # Accept numpy arrays as well
        if isinstance(radar_points, np.ndarray):
            radar_points = torch.from_numpy(radar_points)

        if not torch.is_tensor(radar_points):
            raise TypeError("radar_points must be a torch.Tensor or numpy.ndarray")

        device = radar_points.device
        dtype = radar_points.dtype

        # Ensure we have at least 6 columns
        if radar_points.shape[1] < 6:
            raise ValueError("radar_points must have at least 6 columns: u,v,range,elevation,velocity,power")

        # Get original size for scaling ---
        H_orig, W_orig = int(original_image_size[0]), int(original_image_size[1])

        u = radar_points[:, 0].to(device=device)
        v = radar_points[:, 1].to(device=device)
        rng = radar_points[:, 2].to(device=device)
        elev = radar_points[:, 3].to(device=device)
        vel = radar_points[:, 4].to(device=device)
        power = radar_points[:, 5].to(device=device)

        # Calculate scaling factors
        w_scale = self.W_target / W_orig
        h_scale = self.H_target / H_orig

        # Apply scaling to map original coords to target grid
        u_scaled = u * w_scale
        v_scaled = v * h_scale

        # Convert float coordinates to integer pixel indices for the target grid
        u_idx = torch.clamp(u_scaled.round().long(), 0, self.W_target - 1)
        v_idx = torch.clamp(v_scaled.round().long(), 0, self.H_target - 1)

        # Flattened pixel index for the target grid
        flat_idx = v_idx * self.W_target + u_idx  # shape (N,)

        # Number of bins is now the target grid size
        nbins = self.H_target * self.W_target

        # Compute per-pixel counts
        counts = torch.bincount(flat_idx, minlength=nbins).to(dtype=dtype)

        # Compute sums for each attribute using bincount with weights
        sum_range = torch.bincount(flat_idx, weights=rng.to(dtype=dtype), minlength=nbins).to(dtype=dtype)
        sum_elev = torch.bincount(flat_idx, weights=elev.to(dtype=dtype), minlength=nbins).to(dtype=dtype)
        sum_vel = torch.bincount(flat_idx, weights=vel.to(dtype=dtype), minlength=nbins).to(dtype=dtype)
        sum_power = torch.bincount(flat_idx, weights=power.to(dtype=dtype), minlength=nbins).to(dtype=dtype)

        # Avoid division by zero when computing means
        counts_safe = counts.clone()
        counts_safe[counts_safe == 0] = 1.0

        # Reshape directly to target size ---
        mean_range = (sum_range / counts_safe).reshape(self.H_target, self.W_target)
        mean_elev = (sum_elev / counts_safe).reshape(self.H_target, self.W_target)
        mean_vel = (sum_vel / counts_safe).reshape(self.H_target, self.W_target)
        mean_power = (sum_power / counts_safe).reshape(self.H_target, self.W_target)
        counts_map = counts.reshape(self.H_target, self.W_target)

        # If there were zero counts, set mean channels to zero at those pixels
        zero_mask = (counts_map == 0)
        if zero_mask.any():
            mean_range[zero_mask] = 0.0
            mean_elev[zero_mask] = 0.0
            mean_vel[zero_mask] = 0.0
            mean_power[zero_mask] = 0.0

        img = torch.stack([mean_range, mean_elev, mean_vel, mean_power], dim=0)

        # Ensure float32 for output
        img = img.to(dtype=torch.float32, device=device)

        return img