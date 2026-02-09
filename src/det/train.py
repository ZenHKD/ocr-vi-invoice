"""
DBNet++ Training Script
Train text detection model with TF32 acceleration, tqdm progress, and metrics logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
import csv
from tqdm import tqdm
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.det.dbnet import DBNetPP
from model.det.loss import DBLoss
from src.det.dataloader import create_dataloaders
from src.det.val import validate_epoch


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device: str, epoch: int) -> float:
    """
    Train for one epoch
    
    Args:
        model: DBNet++ model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train DBNet++ text detection model')
    parser.add_argument('--train_dir', type=str, default='data/train_det', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data/val_det', help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=640, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='weights/det', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Enable TF32 for faster training (NO mixed precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(' TF32 enabled for faster training')
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load data
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Model
    model = DBNetPP().to(device)
    print(f'Model: DBNet++ with ResNet50 backbone (disable Deformable Convolution)')
    
    # Loss and optimizer
    criterion = DBLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f'Resumed from epoch {start_epoch} (best F1: {best_f1:.4f})')
    
    # Initialize CSV log
    log_file = save_dir / 'training_log.csv'
    if not log_file.exists() or not args.resume:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_precision', 'val_recall', 'val_f1'])
    
    print(f'\n{"="*80}')
    print('Starting training...')
    print(f'{"="*80}\n')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics (after tqdm is cleared)
        print(f'Epoch: {epoch + 1:3d} | LR: {current_lr:.4f} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Val Precision: {val_metrics["precision"]:.4f} | '
              f'Val Recall: {val_metrics["recall"]:.4f} | '
              f'Val F1: {val_metrics["f1"]:.4f}')
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f'{current_lr:.6f}',
                f'{train_loss:.4f}',
                f'{val_loss:.4f}',
                f'{val_metrics["precision"]:.4f}',
                f'{val_metrics["recall"]:.4f}',
                f'{val_metrics["f1"]:.4f}'
            ])
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f' Saved best model (F1: {best_f1:.4f})')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    print(f'\n{"="*80}')
    print(f'Training completed! Best F1: {best_f1:.4f}')
    print(f'Model saved to: {save_dir / "best_model.pth"}')
    print(f'Training log: {log_file}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
