import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle: RayBundle,
    ):
        # TODO (2): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.arange(self.min_depth, self.max_depth, (self.max_depth- self.min_depth) / self.n_pts_per_ray).view(1, -1, 1).to(ray_bundle.directions.device)
        # TODO (2): Sample points from z values
        sample_points = z_vals * ray_bundle.directions.view(-1, 1, 3) + ray_bundle.origins.view(-1, 1, 3)
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths= z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}