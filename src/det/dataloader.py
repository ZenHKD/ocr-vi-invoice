"""
DBNet++ DataLoader for Text Detection
Loads invoice images with polygon annotations and generates ground truth maps.
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Tuple


class DetectionDataset(Dataset):
    """Dataset for text detection with DBNet++"""
    
    def __init__(self, data_dir: str, image_size: int = 640):
        """
        Args:
            data_dir: Directory containing images and JSON annotations
            image_size: Target image size for training
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Find all JSON annotation files
        self.samples = sorted(list(self.data_dir.glob("*.json")))
        
        # Normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)
    
    def resize_pad(self, image, masks):
        """Resize keeping aspect ratio and pad to target size"""
        h, w = image.shape[:2]
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        if scale != 1.0:
            image = cv2.resize(image, (new_w, new_h))
        
        # Resize masks
        resized_masks = []
        for mask in masks:
            if scale != 1.0:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(mask)
            
        # Create final tensors
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Pad
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        
        # F.pad takes (left, right, top, bottom)
        # Pad bottom/right
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), value=0)
            
        # Normalize
        img_tensor = (img_tensor - self.mean) / self.std
            
        final_masks = []
        for mask in resized_masks:
            mask_tensor = torch.from_numpy(mask).float()
            if pad_h > 0 or pad_w > 0:
                mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=0)
            final_masks.append(mask_tensor.unsqueeze(0))
            
        return img_tensor, final_masks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load image and generate ground truth maps"""
        json_path = self.samples[idx]
        
        # Load annotation
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # Load image
        image_path = self.data_dir / json_path.name.replace('.json', '.jpg')
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract polygons
        polygons = []
        for ann in annotation.get('annotations', []):
            polygon = np.array(ann['polygon'], dtype=np.float32)
            if len(polygon) >= 3:  # Valid polygon
                polygons.append(polygon)
        
        # Generate ground truth maps
        h, w = image.shape[:2]
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        thresh_mask = np.zeros((h, w), dtype=np.float32)
        
        for polygon in polygons:
            # Draw filled polygon for gt
            cv2.fillPoly(gt, [polygon.astype(np.int32)], 1.0)
            
            # Use same polygon for threshold map
            cv2.fillPoly(thresh_map, [polygon.astype(np.int32)], 1.0)
            cv2.fillPoly(thresh_mask, [polygon.astype(np.int32)], 1.0)
        
        # Apply resize and pad
        image_tensor, mask_tensors = self.resize_pad(image, [gt, mask, thresh_map, thresh_mask])
        gt, mask, thresh_map, thresh_mask = mask_tensors
        
        return {
            'image': image_tensor,
            'gt': gt,
            'mask': mask,
            'thresh_map': thresh_map,
            'thresh_mask': thresh_mask
        }


def create_dataloaders(train_dir: str, val_dir: str, batch_size: int = 8, image_size: int = 640, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        batch_size: Batch size for training
        image_size: Image size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    
    # Create datasets
    train_dataset = DetectionDataset(train_dir, image_size=image_size)
    val_dataset = DetectionDataset(val_dir, image_size=image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
