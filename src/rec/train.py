"""
SVTR-CTC Training Script
Train text recognition model with TF32 acceleration, tqdm progress, and metrics logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import math
import os
import sys
import csv
from tqdm import tqdm
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.rec.svtr_ctc import SVTRCTC
from model.rec.loss import CTCLoss
from src.rec.dataloader import create_dataloaders
from src.rec.val import validate_epoch


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, scaler, scheduler, device: str, epoch: int) -> float:
    """
    Train for one epoch
    
    Args:
        model: SVTR-CTC model
        dataloader: Training dataloader
        criterion: Loss function (CTCLoss)
        optimizer: Optimizer
        scaler: GradScaler
        scheduler: LR Scheduler
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
        targets = batch['target'].to(device)
        target_lengths = batch['target_length'].to(device)
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda'):
            log_probs = model(images)  # (T, B, num_classes)
            
            # Input lengths
            input_lengths = batch['input_length'].to(device)
            
            # Compute loss
            log_probs = log_probs.contiguous()
            loss = criterion(log_probs, targets, input_lengths=input_lengths, target_lengths=target_lengths)
        
        # Backward pass with Scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Check scale before step
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        
        # Only step scheduler if optimizer step was successful (scale didn't decrease)
        # If scale decreased, it means gradients were inf/nan and step was skipped
        if new_scale >= old_scale:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train SVTR-CTC text recognition model')
    parser.add_argument('--train_dir', type=str, default='data/train_rec', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data/val_rec', help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=384, help='Image width')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='weights/rec', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Enable TF32 for faster training 
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
        img_size=(args.img_height, args.img_width),
        num_workers=args.num_workers
    )
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Model
    model = SVTRCTC(img_size=(args.img_height, args.img_width)).to(device)
    print(f'Model: SVTR-CTC')
    
    # Loss and optimizer
    criterion = CTCLoss(pad_id=model.tokenizer.pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # FP16 Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Scheduler with OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1000
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f'Resumed from epoch {start_epoch} (best Accuracy: {best_acc:.4f})')
    
    # Initialize CSV log
    log_file = save_dir / 'training_log.csv'
    if not log_file.exists() or not args.resume:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_cer', 'val_accuracy'])
    
    print(f'\n{"="*80}')
    print('Starting training...')
    print(f'{"="*80}\n')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch + 1)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics (after tqdm is cleared)
        print(f'Epoch: {epoch + 1:3d} | LR: {current_lr:.6f} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Val CER: {val_metrics["cer"]:.4f} | '
              f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f'{current_lr:.6f}',
                f'{train_loss:.4f}',
                f'{val_loss:.4f}',
                f'{val_metrics["cer"]:.4f}',
                f'{val_metrics["accuracy"]:.4f}'
            ])
        
        # Save best model based on Accuracy
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f' Saved best model (Accuracy: {best_acc:.4f})')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    print(f'\n{"="*80}')
    print(f'Training completed! Best Accuracy: {best_acc:.4f}')
    print(f'Model saved to: {save_dir / "best_model.pth"}')
    print(f'Training log: {log_file}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
