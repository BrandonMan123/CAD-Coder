#!/usr/bin/env python3
"""
Utility functions for managing pointcloud weights in LLaVA models.
"""

import torch
import os
from typing import Dict, Optional


def save_pointcloud_projector(model, output_path: str):
    """Save only the pointcloud projector weights from a model."""
    if not hasattr(model.get_model(), 'pc_projector'):
        raise ValueError("Model does not have a pointcloud projector")
    
    pc_weights = {}
    for name, param in model.named_parameters():
        if 'pc_projector' in name:
            pc_weights[name] = param.cpu()
    
    torch.save(pc_weights, output_path)
    print(f"Saved pointcloud projector weights to: {output_path}")


def load_pointcloud_projector(model, weights_path: str, strict: bool = False):
    """Load pointcloud projector weights into a model."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    
    pc_weights = torch.load(weights_path, map_location='cpu')
    
    # Filter weights to only pc_projector components
    filtered_weights = {k: v for k, v in pc_weights.items() if 'pc_projector' in k}
    
    missing_keys, unexpected_keys = model.load_state_dict(filtered_weights, strict=strict)
    
    print(f"Loaded pointcloud weights from: {weights_path}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    return missing_keys, unexpected_keys


def check_pointcloud_compatibility(model) -> Dict[str, bool]:
    """Check if a model has pointcloud support and what components are available."""
    result = {
        'has_pointcloud_projector': hasattr(model.get_model(), 'pc_projector'),
        'has_encode_pointclouds': hasattr(model, 'encode_pointclouds'),
        'config_has_pointcloud': getattr(model.config, 'use_pointcloud', False),
        'pc_projector_type': getattr(model.config, 'pc_projector_type', None),
        'pointcloud_hidden_size': getattr(model.config, 'pointcloud_hidden_size', None),
    }
    return result


def initialize_pointcloud_from_pretrained(model, pretrained_path: str):
    """Initialize pointcloud components using pretrained weights."""
    compatibility = check_pointcloud_compatibility(model)
    
    if not compatibility['has_pointcloud_projector']:
        print("Model doesn't have pointcloud projector. Initializing...")
        # This would require calling initialize_pointcloud_modules
        return False
    
    load_pointcloud_projector(model, pretrained_path)
    return True