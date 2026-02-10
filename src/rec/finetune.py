"""
ResNet-CTC Finetuning Script
Finetune text recognition model on real data with TF32 acceleration, tqdm progress, and metrics logging.
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

from model.rec.resnet_ctc import ResNetCTC
from model.rec.loss import CTCLoss
from src.rec.dataloader import create_dataloaders
from src.rec.val import validate_epoch
from src.rec.train import train_epoch

def main():
    parser = argparse.ArgumentParser(description='Finetune ResNet-CTC text recognition model')
    parser.add_argument('--train_dir', type=str, default='data/archive/text_recognition_mcocr_data/text_recognition_mcocr_data', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data/archive/text_recognition_mcocr_data/text_recognition_mcocr_data', help='Validation data directory')
    parser.add_argument('--train_labels', type=str, default='labels_train.csv', help='Training labels CSV file')
    parser.add_argument('--val_labels', type=str, default='labels_val.csv', help='Validation labels CSV file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (lower for finetuning)')
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=256, help='Image width')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'], help='Backbone network')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='weights/rec_finetune', help='Directory to save checkpoints')
    parser.add_argument('--pretrained_path', type=str, default='weights/rec/best_model.pth', help='Path to pre-trained model (ignored if resume is set)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Enable TF32 for faster training 
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('TF32 enabled for faster training')
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load data
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width),
        num_workers=args.num_workers,
        train_labels_file=args.train_labels,
        val_labels_file=args.val_labels
    )
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Model
    model = ResNetCTC(name=args.backbone, in_channels=3).to(device)
    print(f'Model: ResNet-CTC ({args.backbone})')
    
    # Loss and optimizer
    criterion = CTCLoss(pad_id=model.tokenizer.pad_id, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    start_epoch = 0
    best_acc = 0.0
    
    # Load Checkpoint (Resume or Pretrained)
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        # scaler state dict? The training loop re-inits scaler, but usually we should load it.
        # For simplicity in this finetuning script, we'll re-init scaler as it adjusts quickly.
        print(f'Resumed from epoch {start_epoch} (best Accuracy: {best_acc:.4f})')
        
    elif args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pre-trained weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # Determine if we are loading a full checkpoint or just state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    else:
        print(f"Training from scratch (no checkpoint loaded).")
    
    # FP16 Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Scheduler with OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs, # Note: OneCycleLR might behave oddly if resuming mid-cycle without loading scheduler state. 
                            # But since we are likely extending epochs (e.g. 50 -> 100), we probably want a NEW cycle or to adjust.
                            # For simple finetuning resumption, we'll start a new cycle or let it be.
                            # Ideally, we should adjust total_steps if resuming to finish a cycle, 
                            # but here we usually resume to *add* epochs.
                            # Let's keep it simple: restart scheduler for the remaining epochs or full epochs?
                            # Use args.epochs as TOTAL epochs if we strictly followed pytorch logic, 
                            # but users often treat --epochs as "more epochs".
                            # Let's assume args.epochs is the target TOTAL epochs.
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000,
        last_epoch=start_epoch * len(train_loader) - 1 if start_epoch > 0 else -1
    )
    
    # Initialize CSV log
    log_file = save_dir / 'finetune_log.csv'
    # If resuming, we append. If not resuming, we overwrite (unless pre-existing? safe to append if file exists)
    if not args.resume or not log_file.exists():
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_cer', 'val_accuracy'])
    
    print(f'\n{"="*80}')
    print('Starting finetuning...')
    print(f'{"="*80}\n')
    
    if start_epoch == 0:
        # Initial validation only if starting fresh
        print("Running initial validation...")
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        print(f'Initial | Val Loss: {val_loss:.4f} | Val CER: {val_metrics["cer"]:.4f} | Val Accuracy: {val_metrics["accuracy"]:.4f}')
        if not args.resume:
             best_acc = val_metrics["accuracy"]

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
        
    
    print(f'\n{"="*80}')
    print(f'Finetuning completed! Best Accuracy: {best_acc:.4f}')
    print(f'Model saved to: {save_dir / "best_model.pth"}')
    print(f'Training log: {log_file}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
