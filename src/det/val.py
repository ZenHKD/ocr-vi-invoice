"""
DBNet++ Validation Script
Evaluates text detection model with precision, recall, and F1 metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


def compute_metrics(pred_binary: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for text detection
    
    Args:
        pred_binary: Predicted binary map (N, 1, H, W), values in [0, 1]
        gt: Ground truth binary map (N, 1, H, W)  
        mask: Valid region mask (N, 1, H, W)
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Threshold predictions at 0.5
    pred_binary = (pred_binary > 0.5).float()
    
    # Apply mask
    pred_binary = pred_binary * mask
    gt = gt * mask
    
    # Compute TP, FP, FN
    tp = ((pred_binary == 1) & (gt == 1)).float().sum()
    fp = ((pred_binary == 1) & (gt == 0)).float().sum()
    fn = ((pred_binary == 0) & (gt == 1)).float().sum()
    
    # Compute metrics
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def validate_epoch(model: nn.Module, dataloader, criterion, device: str) -> Tuple[float, Dict[str, float]]:
    """
    Run validation for one epoch
    
    Args:
        model: DBNet++ model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        (avg_loss, metrics) where metrics contains precision, recall, F1
    """
    model.eval()
    
    total_loss = 0
    all_precision = []
    all_recall = []
    all_f1 = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            # Move to device
            images = batch['image'].to(device)
            gt = batch['gt'].to(device)
            mask = batch['mask'].to(device)
            thresh_map = batch['thresh_map'].to(device)
            thresh_mask = batch['thresh_mask'].to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            batch_dict = {
                'gt': gt,
                'mask': mask,
                'thresh_map': thresh_map,
                'thresh_mask': thresh_mask
            }
            loss, loss_dict = criterion(predictions, batch_dict)
            total_loss += loss.item()
            
            # Compute metrics
            pred_binary = predictions['binary']
            metrics = compute_metrics(pred_binary, gt, mask)
            
            all_precision.append(metrics['precision'])
            all_recall.append(metrics['recall'])
            all_f1.append(metrics['f1'])
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1': np.mean(all_f1)
    }
    
    return avg_loss, avg_metrics


def main():
    """Standalone validation script"""
    import argparse
    import sys
    import os
    
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from model.det.dbnet import DBNetPP
    from model.det.loss import DBLoss
    from src.det.dataloader import create_dataloaders
    
    parser = argparse.ArgumentParser(description='Validate DBNet++ model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val_dir', type=str, default='data/val_det', help='Validation data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=640, help='Image size')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load model
    model = DBNetPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint} (Epoch {checkpoint["epoch"]})')
    
    # Load data
    _, val_loader = create_dataloaders(
        train_dir=args.val_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Loss criterion
    criterion = DBLoss()
    
    # Validate
    val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
    
    print(f'\n{"="*50}')
    print(f'Validation Results:')
    print(f'{"="*50}')
    print(f'Loss:      {val_loss:.4f}')
    print(f'Precision: {val_metrics["precision"]:.4f}')
    print(f'Recall:    {val_metrics["recall"]:.4f}')
    print(f'F1 Score:  {val_metrics["f1"]:.4f}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
