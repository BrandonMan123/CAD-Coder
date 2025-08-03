import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_pointcloud_projector(config, delay_load=False, **kwargs):
    """
    Build pointcloud projector to map pointcloud features to language model hidden size.
    Pointcloud input: (batch_size, num_tokens, pointcloud_hidden_size) 
    Output: (batch_size, num_tokens, hidden_size)
    """
    projector_type = getattr(config, 'pc_projector_type', 'linear')
    pointcloud_hidden_size = getattr(config, 'pointcloud_hidden_size', 382)
    hidden_size = config.hidden_size

    if projector_type == 'linear':
        return nn.Linear(pointcloud_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(pointcloud_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        # For identity, we need to ensure dimension compatibility
        if pointcloud_hidden_size == hidden_size:
            return IdentityMap()
        else:
            # Add a linear layer to match dimensions
            return nn.Linear(pointcloud_hidden_size, hidden_size)

    # Pointcloud-specific projector types
    if projector_type == 'mlp2x_gelu':
        return nn.Sequential(
            nn.Linear(pointcloud_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    if projector_type == 'resblock':
        # Use a residual block for better feature learning
        return nn.Sequential(
            nn.Linear(pointcloud_hidden_size, hidden_size),
            SimpleResBlock(hidden_size)
        )

    raise ValueError(f'Unknown pointcloud projector type: {projector_type}')
