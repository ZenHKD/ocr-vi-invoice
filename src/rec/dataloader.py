"""
ResNet-CTC DataLoader for Text Recognition
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
    """Dataset for text recognition with ResNet-CTC"""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (32, 256), labels_file: str = 'labels.csv'):
        """
        Args:
            data_dir: Directory containing images and labels.txt
            img_size: Target image size (height, width)
            labels_file: Name of the labels CSV file
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(VOCAB)
        
        # Load labels from labels.csv
        self.samples = []
        labels_path = self.data_dir / labels_file
        
        df = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        self.samples = list(zip(df['filename'], df['text']))
        
        # Normalization constants (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)
    
    def resize_pad(self, image):
        """Resize to fixed size (32, 256)"""
        h, w = image.shape[:2]
        target_h, target_w = self.img_size 
        
        # 1. Resize height to 32
        scale = target_h / h
        new_h = target_h
        new_w = int(w * scale)
        
        # 2. Check width
        if new_w > target_w:
            # Resize directly to target_w (32x256) - distort width
            image = cv2.resize(image, (target_w, target_h))
        else:
            # Resize height, keep aspect ratio width
            image = cv2.resize(image, (new_w, new_h))
            # Pad width with white pixels (255)
            pad_w = target_w - new_w
            if pad_w > 0:
                # Pad right side
                image = cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
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
            # Fallback
            print(f"Warning: Failed to load {img_path}")
            image = np.ones((32, 256, 3), dtype=np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode text 
        target = self.tokenizer.encode([text])[0]
        target_length = len(target)
        
        # Resize to fixed size
        image_tensor = self.resize_pad(image)
        
        return {
            'image': image_tensor,
            'target': target,
            'target_length': target_length,
            'text': text
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for fixed size images
    """
    texts = [item['text'] for item in batch]
    
    # 1. Handle Images (Fixed 32x256)
    images = torch.stack([item['image'] for item in batch])
    
    # Input length is fixed for CTC based on width
    # ResNet with strides (2, 2), (2, 2), (2, 1), (2, 1) -> W / 4
    # 256 / 4 = 64
    batch_size = images.shape[0]
    fixed_width = images.shape[3] # 256
    input_length_val = fixed_width // 4
    input_lengths = torch.full((batch_size,), input_length_val, dtype=torch.long)
    
    # 2. Handle Targets
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    targets_raw = [item['target'] for item in batch]
    targets = pad_sequence(targets_raw, batch_first=True, padding_value=1) # 1 is pad_id assumed (check Tokenizer)
    
    return {
        'image': images,
        'target': targets,
        'target_length': target_lengths,
        'input_length': input_lengths,
        'text': texts
    }


def create_dataloaders(train_dir: str, val_dir: str, batch_size: int = 16, 
                       img_size: Tuple[int, int] = (32, 256),
                       num_workers: int = 4,
                       train_labels_file: str = 'labels.csv',
                       val_labels_file: str = 'labels.csv') -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        batch_size: Batch size for training
        img_size: Image size (height, width)
        num_workers: Number of workers for data loading
        train_labels_file: Labels CSV filename for training
        val_labels_file: Labels CSV filename for validation
    
    Returns:
        train_loader, val_loader
    """
    
    # Create datasets
    train_dataset = RecognitionDataset(train_dir, img_size=img_size, labels_file=train_labels_file)
    val_dataset = RecognitionDataset(val_dir, img_size=img_size, labels_file=val_labels_file)
    
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
