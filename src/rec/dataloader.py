"""
SVTR-CTC DataLoader for Text Recognition
Loads cropped text images with labels from labels.txt file.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.rec.tokenizer import Tokenizer
from model.rec.vocab import VOCAB


class RecognitionDataset(Dataset):
    """Dataset for text recognition with SVTR-CTC"""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (32, 128)):
        """
        Args:
            data_dir: Directory containing images and labels.txt
            img_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(VOCAB)
        
        # Load labels from labels.csv
        self.samples = []
        labels_file = self.data_dir / 'labels.csv'
        
        df = pd.read_csv(labels_file, dtype=str, keep_default_na=False)
        self.samples = list(zip(df['filename'], df['text']))
        
        # Normalization constants (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)
    
    def resize_pad(self, image, min_width=0):
        """Resize with fixed height and variable width (maintaining aspect ratio)"""
        h, w = image.shape[:2]
        target_h, target_w = self.img_size  # (32, 128) - 128 is just a base/max reference now
        
        # Scale to fixed height = 32
        scale = target_h / h
        new_h, new_w = int(h * scale), int(w * scale)
        if new_w % 4 != 0:
            new_w = ((new_w // 4) + 1) * 4
        
        # Enforce minimum width if specified (for CTC stability)
        if new_w < min_width:
            new_w = min_width
            # Ensure divisibility by 4
            if new_w % 4 != 0:
                new_w = ((new_w // 4) + 1) * 4

        # Resize
        image = cv2.resize(image, (new_w, new_h))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Normalize
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load image and encode text label"""
        img_name, text = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / img_name
        
        # For Vietnamese/Unicode paths
        try:
            np_img = np.fromfile(str(img_path), np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError
        except Exception:
            # Fallback or error handling
            # Create a dummy white image if failed (to avoid crashing)
            print(f"Warning: Failed to load {img_path}")
            image = np.ones((32, 128, 3), dtype=np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode text first to determine minimum required width
        target = self.tokenizer.encode([text])[0]
        # Use length of ENCODED target, not original text
        target_length = len(target)
        
        # Minimum width required by CTC (input_length >= target_length)
        # downsample factor is 4, so width must be at least 4 * target_length
        min_width = target_length * 4
        
        # Resize (variable width) with constraint
        image_tensor = self.resize_pad(image, min_width=min_width)
        
        return {
            'image': image_tensor,
            'target': target,
            'target_length': target_length,
            'text': text
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences and images
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary with padded targets and images
    """
    texts = [item['text'] for item in batch]
    
    # 1. Handle Images (Variable Width)
    # Find max width in this batch
    max_w = max(item['image'].shape[2] for item in batch)
    
    # Ensure max_w is divisible by 4 (network stride requirements)
    if max_w % 4 != 0:
        max_w = ((max_w // 4) + 1) * 4
        
    padded_images = []
    input_lengths = []
    for item in batch:
        img = item['image'] # (C, H, W)
        c, h, w = img.shape
        
        # Calculate valid input length for CTC (W / 4)
        input_lengths.append(w // 4)
        
        pad_w = max_w - w
        if pad_w > 0:
            # F.pad takes (left, right, top, bottom)
            # Usually padding with 0 is fine for normalized images.
            img = F.pad(img, (0, pad_w, 0, 0), value=0)
        padded_images.append(img)
        
    images = torch.stack(padded_images)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    
    # 2. Handle Targets
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    targets_raw = [item['target'] for item in batch]
    targets = pad_sequence(targets_raw, batch_first=True, padding_value=1) # 1 is pad_id
    
    return {
        'image': images,
        'target': targets,
        'target_length': target_lengths,
        'input_length': input_lengths,
        'text': texts
    }


def create_dataloaders(train_dir: str, val_dir: str, batch_size: int = 16, 
                       img_size: Tuple[int, int] = (32, 384),
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        batch_size: Batch size for training
        img_size: Image size (height, width)
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    
    # Create datasets
    train_dataset = RecognitionDataset(train_dir, img_size=img_size)
    val_dataset = RecognitionDataset(val_dir, img_size=img_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader
