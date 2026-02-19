"""
DBNet++ Training Script
Train text detection model with:
  - AMP (mixed precision), OneCycleLR scheduler, gradient clipping
  - TF32 acceleration, tqdm progress, and metrics logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
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


def train_epoch(model: nn.Module, dataloader, criterion, optimizer,
                scaler, scheduler, device: str, epoch: int) -> float:
    """
    Train for one epoch with AMP + gradient clipping.

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

        # Forward pass with AMP
        with torch.amp.autocast('cuda'):
            predictions = model(images)

            batch_dict = {
                'gt': gt,
                'mask': mask,
                'thresh_map': thresh_map,
                'thresh_mask': thresh_mask
            }
            loss, loss_dict = criterion(predictions, batch_dict)

        if not torch.isfinite(loss):
            print(f"  Warning: NaN/Inf loss detected. Skipping step.")
            continue

        # Backward pass with scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Optimizer step
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()

        # Only step scheduler if optimizer step was successful
        if new_scale >= old_scale:
            scheduler.step()

        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train DBNet++ text detection model')
    parser.add_argument('--train_dir', type=str, default='data/train_det', help='Training data directory (synthetic)')
    parser.add_argument('--val_dir', type=str, default='data/val_det_sroie', help='Validation data directory (SROIE train)')
    parser.add_argument('--test_dir', type=str, default='data/test_det_sroie', help='Test data directory (SROIE test)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--freeze_epochs', type=int, default=5, help='Freeze backbone for first N epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Max learning rate')
    parser.add_argument('--image_size', type=int, default=960, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='weights/det', help='Directory to save checkpoints')
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

    # Load data (3-way split: synthetic train, SROIE val, SROIE test)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # Model
    model = DBNetPP().to(device)
    print(f'Model: DBNet++ with ResNet50 backbone (enable Deformable Convolution)')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')

    # Freeze backbone for first N epochs
    def freeze_backbone():
        for p in model.backbone.parameters():
            p.requires_grad = False
        frozen = sum(1 for p in model.backbone.parameters())
        print(f'  ❄️  Backbone frozen ({frozen} param groups)')

    def unfreeze_backbone():
        for p in model.backbone.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  🔥 Backbone unfrozen — trainable params: {trainable:,}')

    if args.freeze_epochs > 0:
        freeze_backbone()

    # Loss
    criterion = DBLoss()

    # Optimizer — only trainable params
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.05
    )

    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda')

    # OneCycleLR scheduler (per-batch stepping)
    remaining_epochs = args.epochs
    if args.freeze_epochs > 0:
        remaining_epochs = args.freeze_epochs  # first schedule covers freeze phase
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=remaining_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.075,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000
    )

    # Resume from checkpoint if specified
    best_f1 = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        prev_epoch = checkpoint.get('epoch', '?')
        prev_f1 = checkpoint.get('best_f1', 0.0)
        print(f'Loaded weights from {args.resume}')
        print(f'  Previous training: epoch {prev_epoch}, F1 {prev_f1:.4f}')
        print(f'  Starting fresh optimizer + LR schedule for continued training')

    # Initialize CSV log
    log_file = save_dir / 'training_log.csv'
    if not log_file.exists() or not args.resume:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_iou', 'val_dice'])

    print(f'\n{"="*80}')
    print('Starting training...')
    print(f'{"="*80}\n')

    # Training loop
    for epoch in range(args.epochs):
        # Unfreeze backbone after freeze phase
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            unfreeze_backbone()
            # Differential LR: backbone gets 10x lower LR to preserve pretrained features
            backbone_params = list(model.backbone.parameters())
            other_params = [p for n, p in model.named_parameters() if not n.startswith('backbone')]
            finetune_lr = args.lr * 0.5  # lower overall max LR for stability
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': finetune_lr / 10},  # backbone: very gentle
                {'params': other_params, 'lr': finetune_lr},          # neck+head: normal
            ], weight_decay=0.05)
            scaler = torch.amp.GradScaler('cuda')
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[finetune_lr / 10, finetune_lr],  # per param-group max LR
                epochs=args.epochs - args.freeze_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,  # longer warmup for stability
                anneal_strategy='cos',
                div_factor=10,   # gentler start (LR/10 instead of LR/25)
                final_div_factor=1000
            )
            print(f'  Differential LR: backbone={finetune_lr/10:.6f}, head={finetune_lr:.6f}')

        # Train with AMP
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, scheduler, device, epoch + 1
        )

        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        phase = 'freeze' if epoch < args.freeze_epochs else 'finetune'

        # Print metrics
        print(f'Epoch: {epoch + 1:3d} [{phase:8s}] | LR: {current_lr:.6f} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'P: {val_metrics["precision"]:.4f} | '
              f'R: {val_metrics["recall"]:.4f} | '
              f'F1: {val_metrics["f1"]:.4f} | '
              f'IoU: {val_metrics["iou"]:.4f} | '
              f'Dice: {val_metrics["dice"]:.4f}')

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
                f'{val_metrics["f1"]:.4f}',
                f'{val_metrics["iou"]:.4f}',
                f'{val_metrics["dice"]:.4f}',
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
            print(f'  → Saved best model (F1: {best_f1:.4f})')

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
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
    print(f'{"="*80}')

    # Final test evaluation
    if test_loader:
        print(f'\nRunning test evaluation with best model...')
        best_ckpt = torch.load(save_dir / 'best_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
        print(f'\n{"="*50}')
        print(f'Test Results (SROIE):')
        print(f'{"="*50}')
        print(f'Loss:      {test_loss:.4f}')
        print(f'Precision: {test_metrics["precision"]:.4f}')
        print(f'Recall:    {test_metrics["recall"]:.4f}')
        print(f'F1 Score:  {test_metrics["f1"]:.4f}')
        print(f'IoU:       {test_metrics["iou"]:.4f}')
        print(f'Dice:      {test_metrics["dice"]:.4f}')
        print(f'{"="*50}')


if __name__ == '__main__':
    main()
