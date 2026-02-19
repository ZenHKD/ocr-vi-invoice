"""
SVTRv2 Validation Script
Evaluates text recognition model with CER and Accuracy metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm
import editdistance


def compute_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute Character Error Rate (CER) using edit distance."""
    total_chars = 0
    total_errors = 0

    for pred, gt in zip(predictions, ground_truths):
        errors = editdistance.eval(pred, gt)
        total_errors += errors
        total_chars += len(gt)

    return total_errors / max(total_chars, 1)


def compute_acc(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute Accuracy (exact match percentage)."""
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / max(len(predictions), 1)


def validate_epoch(model: nn.Module, dataloader, criterion, device: str) -> Tuple[float, Dict[str, float]]:
    """
    Run validation for one epoch.

    Args:
        model: SVTRv2 model
        dataloader: Validation dataloader
        criterion: Loss function (SVTRv2Loss)
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
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            target_lengths = batch['target_length'].to(device)
            texts = batch['text']

            # Forward pass (eval mode — no SGM)
            log_probs = model(images)  # (T, B, num_classes)

            # Input lengths
            input_lengths = batch['input_length'].to(device)

            # Compute CTC-only loss for validation
            loss = criterion(log_probs, targets,
                             input_lengths=input_lengths,
                             target_lengths=target_lengths)
            total_loss += loss.item()

            # Decode predictions
            predictions = model.decode_probs(log_probs)

            all_predictions.extend(predictions)
            all_ground_truths.extend(texts)

    avg_loss = total_loss / max(len(dataloader), 1)

    cer = compute_cer(all_predictions, all_ground_truths)
    accuracy = compute_acc(all_predictions, all_ground_truths)

    avg_metrics = {
        'cer': cer,
        'accuracy': accuracy
    }

    return avg_loss, avg_metrics


def main():
    """Standalone validation script."""
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from model.rec2.svtrv2 import SVTRv2
    from model.rec2.loss import SVTRv2Loss
    from src.rec2.dataloader import create_dataloaders

    parser = argparse.ArgumentParser(description='Validate SVTRv2 model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val_annotation', type=str,
                        default='data/archive/text_recognition_train_data.txt',
                        help='MCOCR validation annotation file')
    parser.add_argument('--val_img_dir', type=str,
                        default='data/archive/text_recognition_mcocr_data/text_recognition_mcocr_data',
                        help='MCOCR image directory')
    parser.add_argument('--test_annotation', type=str, default=None,
                        help='MCOCR test annotation file (optional)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=256, help='Image width')
    parser.add_argument('--variant', type=str, default='small', choices=['tiny', 'small', 'base'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model
    model = SVTRv2(variant=args.variant).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint} (Epoch {checkpoint["epoch"] + 1})')

    # Load data
    _, val_loader, test_loader = create_dataloaders(
        vietocr_dir='data/vietocr',
        val_annotation=args.val_annotation,
        val_img_dir=args.val_img_dir,
        test_annotation=args.test_annotation,
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width),
    )

    criterion = SVTRv2Loss()

    # Validate
    val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

    print(f'\n{"="*50}')
    print(f'Validation Results:')
    print(f'{"="*50}')
    print(f'Loss:      {val_loss:.4f}')
    print(f'CER:       {val_metrics["cer"]:.4f}')
    print(f'Accuracy:  {val_metrics["accuracy"]:.4f}')
    print(f'{"="*50}')

    # Test if available
    if test_loader:
        test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
        print(f'\n{"="*50}')
        print(f'Test Results:')
        print(f'{"="*50}')
        print(f'Loss:      {test_loss:.4f}')
        print(f'CER:       {test_metrics["cer"]:.4f}')
        print(f'Accuracy:  {test_metrics["accuracy"]:.4f}')
        print(f'{"="*50}')


if __name__ == '__main__':
    main()
