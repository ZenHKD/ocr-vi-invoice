"""
SVTR-CTC Validation Script
Evaluates text recognition model with CER and Accuracy metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm
import editdistance


def compute_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute Character Error Rate (CER) using edit distance
    
    Args:
        predictions: List of predicted strings
        ground_truths: List of ground truth strings
    
    Returns:
        Character Error Rate (CER)
    """
    total_chars = 0
    total_errors = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Calculate edit distance
        errors = editdistance.eval(pred, gt)
        total_errors += errors
        total_chars += len(gt)
    
    cer = total_errors / max(total_chars, 1)
    return cer


def compute_acc(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute Accuracy (percentage of samples with 100% correct character prediction)
    
    Args:
        predictions: List of predicted strings
        ground_truths: List of ground truth strings
    
    Returns:
        Accuracy (0.0 to 1.0)
    """
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    accuracy = correct / len(predictions)
    return accuracy


def validate_epoch(model: nn.Module, dataloader, criterion, device: str) -> Tuple[float, Dict[str, float]]:
    """
    Run validation for one epoch
    
    Args:
        model: SVTR-CTC model
        dataloader: Validation dataloader
        criterion: Loss function (CTCLoss)
        device: Device to run on
    
    Returns:
        (avg_loss, metrics) where metrics contains CER and Accuracy
    """
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            # Move to device
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            target_lengths = batch['target_length'].to(device)
            texts = batch['text']
            
            # Forward pass
            log_probs = model(images)  # (T, B, num_classes)
            
            # Input lengths
            input_lengths = batch['input_length'].to(device)
            
            # Compute loss
            loss = criterion(log_probs, targets, input_lengths=input_lengths, target_lengths=target_lengths)
            total_loss += loss.item()
            
            # Decode predictions
            predictions = model.decode_probs(log_probs)
            
            # Collect for metrics
            all_predictions.extend(predictions)
            all_ground_truths.extend(texts)
    
    # Average loss
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    cer = compute_cer(all_predictions, all_ground_truths)
    accuracy = compute_acc(all_predictions, all_ground_truths)
    
    avg_metrics = {
        'cer': cer,
        'accuracy': accuracy
    }
    
    return avg_loss, avg_metrics


def main():
    """Standalone validation script"""
    import argparse
    import sys
    import os
    
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from model.rec.svtr_ctc import SVTRCTC
    from model.rec.loss import CTCLoss
    from src.rec.dataloader import create_dataloaders
    
    parser = argparse.ArgumentParser(description='Validate SVTR-CTC model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val_dir', type=str, default='data/val_rec', help='Validation data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=128, help='Image width')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load model
    model = SVTRCTC(img_size=(args.img_height, args.img_width)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint} (Epoch {checkpoint["epoch"] + 1})')
    
    # Load data
    _, val_loader = create_dataloaders(
        train_dir=args.val_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width)
    )
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Loss criterion
    criterion = CTCLoss()
    
    # Validate
    val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
    
    print(f'\n{"="*50}')
    print(f'Validation Results:')
    print(f'{"="*50}')
    print(f'Loss:      {val_loss:.4f}')
    print(f'CER:       {val_metrics["cer"]:.4f}')
    print(f'Accuracy:  {val_metrics["accuracy"]:.4f}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
