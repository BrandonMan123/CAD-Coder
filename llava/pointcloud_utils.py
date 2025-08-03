"""
Pointcloud data loading utilities for LLaVA.
"""

import torch
import os
from typing import List, Union, Optional
import numpy as np


def load_pointcloud_tensor(file_path: str, device: str = 'cpu') -> torch.Tensor:
    """
    Load pointcloud tensor from .pt file with validation.
    
    Args:
        file_path: Path to .pt file containing pointcloud tensor
        device: Device to load tensor on
        
    Returns:
        torch.Tensor of shape (64, 382)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pointcloud file not found: {file_path}")
    
    try:
        pointcloud = torch.load(file_path, map_location=device)
        
        # Validate shape
        if not isinstance(pointcloud, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(pointcloud)}")
        
        if pointcloud.shape != (64, 382):
            raise ValueError(f"Expected shape (64, 382), got {pointcloud.shape}")
            
        return pointcloud.float()
        
    except Exception as e:
        raise ValueError(f"Failed to load pointcloud from {file_path}: {e}")


def batch_load_pointclouds(file_paths: List[str], device: str = 'cpu') -> torch.Tensor:
    """
    Load multiple pointcloud files and stack into batch tensor.
    
    Args:
        file_paths: List of paths to .pt files
        device: Device to load tensors on
        
    Returns:
        torch.Tensor of shape (batch_size, 64, 382)
    """
    pointclouds = []
    for file_path in file_paths:
        pc = load_pointcloud_tensor(file_path, device)
        pointclouds.append(pc)
    
    return torch.stack(pointclouds, dim=0)


def create_sample_pointcloud(output_path: str, seed: Optional[int] = 42) -> torch.Tensor:
    """Create a sample pointcloud tensor for testing."""
    if seed is not None:
        torch.manual_seed(seed)
    
    pointcloud = torch.randn(64, 382)
    torch.save(pointcloud, output_path)
    return pointcloud