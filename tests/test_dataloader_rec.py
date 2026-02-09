"""
Test script to visualize dataloader preprocessing
Verifies that images are correctly resized to 32xW (dynamic width) and batched correctly.
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rec.dataloader import RecognitionDataset


def denormalize(img_tensor):
    """Denormalize image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()
    return img_np


def test_dataloader_visualization():
    """Visualize preprocessing results"""
    
    # Load dataset
    print("Loading dataset from data/val_rec...")
    dataset = RecognitionDataset('data/val_rec', img_size=(32, 128))
    
    print(f"Total samples: {len(dataset)}")
    
    # Get first few samples
    num_samples = min(6, len(dataset))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nVisualizing {num_samples} samples...\n")
    
    for idx in range(num_samples):
        sample = dataset[idx]
        
        # Get original image
        img_name, text = dataset.samples[idx]
        img_path = dataset.data_dir / img_name
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Get preprocessed image
        preprocessed_img = denormalize(sample['image'])
        
        # Plot original
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original: {original_img.shape[1]}x{original_img.shape[0]}\n"{text[:30]}"', 
                               fontsize=10)
        axes[idx, 0].axis('off')
        
        # Plot preprocessed
        axes[idx, 1].imshow(preprocessed_img)
        preprocessed_shape = sample['image'].shape
        axes[idx, 1].set_title(f'Preprocessed: {preprocessed_shape[2]}x{preprocessed_shape[1]}\n'
                               f'(C, H, W) = {preprocessed_shape}', 
                               fontsize=10)
        axes[idx, 1].axis('off')
        
        # Print info
        print(f"Sample {idx}:")
        print(f"  File: {img_name}")
        print(f"  Text: {text}")
        print(f"  Original shape (H, W, C): {original_img.shape}")
        print(f"  Preprocessed shape (C, H, W): {preprocessed_shape}")
        print(f"  Height = {preprocessed_shape[1]}, Width = {preprocessed_shape[2]}")
        print()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('tests/dataloader_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Saved visualization to: {output_path}")
    
    plt.show()
    
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION (Single Sample):")
    print("="*60)
    sample = dataset[0]
    C, H, W = sample['image'].shape
    print(f"Tensor shape: (C={C}, H={H}, W={W})")
    print(f" Height = {H} (expected: 32)")
    print(f" Width = {W} (variable, should be multiple of 4)")
    
    if H == 32 and W % 4 == 0:
        print(f" SUCCESS: Image is correctly resized to 32x{W}")
    else:
        print(f" ERROR: Expected (32, multiple of 4) but got ({H}, {W})")
        
    # Verify Collate Function (Batching)
    print("\n" + "="*60)
    print("VERIFICATION (Batch/Collate):")
    print("="*60)
    
    from src.rec.dataloader import collate_fn
    from torch.utils.data import DataLoader
    
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))
    
    images = batch['image']
    input_lengths = batch.get('input_length')
    
    print(f"Batch Shape: {images.shape}")
    
    if input_lengths is not None:
        print(f"Input Lengths: {input_lengths}")
        print(" SUCCESS: 'input_length' key exists in batch.")
        
        # Verify values
        B, C, H, W_batch = images.shape
        expected_T = W_batch // 4
        
        # input_lengths should be <= expected_T
        if (input_lengths <= expected_T).all():
             print(f" SUCCESS: All input_lengths are <= {expected_T} (Batch Width/4)")
        else:
             print(f" ERROR: Some input_lengths > {expected_T}!")
    else:
        print(" ERROR: 'input_length' key MISSING in batch!")

    print("="*60)


if __name__ == '__main__':
    test_dataloader_visualization()
