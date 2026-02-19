"""
SVTRv2 Training Script
Train text recognition model with:
  - SVTRv2 backbone + FRM + CTC head
  - SGM (Semantic Guidance Module) for training-only linguistic supervision
  - AMP (mixed precision), OneCycleLR scheduler, gradient clipping
  - CER / Accuracy validation metrics
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

from model.rec2.svtrv2 import SVTRv2
from model.rec2.loss import SVTRv2Loss
from src.rec2.dataloader import create_dataloaders
from src.rec2.val import validate_epoch


def train_epoch(model: nn.Module, dataloader, criterion, optimizer,
                scaler, scheduler, device: str, epoch: int) -> float:
    """
    Train for one epoch with SGM.

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        target_lengths = batch['target_length'].to(device)
        input_lengths = batch['input_length'].to(device)

        # Forward pass with AMP
        with torch.amp.autocast('cuda'):
            # Training mode: returns (log_probs, sgm_output)
            result = model(images, targets=targets)

            if isinstance(result, tuple):
                log_probs, sgm_output = result
            else:
                log_probs = result
                sgm_output = None

            log_probs = log_probs.contiguous()

            # Combined loss: CTC + SGM
            loss = criterion(
                log_probs, targets, sgm_output=sgm_output,
                input_lengths=input_lengths, target_lengths=target_lengths
            )

        if not torch.isfinite(loss):
            print(f"  Warning: NaN/Inf loss detected. Skipping step.")
            continue

        # Backward pass
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

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train SVTRv2 text recognition model')

    # Data paths
    parser.add_argument('--vietocr_dir', type=str, default='data/vietocr',
                        help='VietOCR data root directory')
    parser.add_argument('--val_annotation', type=str,
                        default='data/archive/text_recognition_train_data.txt',
                        help='MCOCR validation annotation file')
    parser.add_argument('--val_img_dir', type=str,
                        default='data/archive/text_recognition_mcocr_data/text_recognition_mcocr_data',
                        help='MCOCR image directory for validation')
    parser.add_argument('--test_annotation', type=str,
                        default='data/archive/text_recognition_val_data.txt',
                        help='MCOCR test annotation file')

    # Training config
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00065, help='Max learning rate')
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=256, help='Image width')
    parser.add_argument('--variant', type=str, default='base', choices=['tiny', 'small', 'base'],
                        help='SVTRv2 variant')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loading workers')
    parser.add_argument('--save_dir', type=str, default='weights/rec2', help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (loads model weights only)')
    parser.add_argument('--lambda_sgm', type=float, default=0.1, help='SGM loss weight')
    parser.add_argument('--augment', action='store_true', help='Enable heavy data augmentation')
    args = parser.parse_args()

    # Create save directory (use separate dir for augmented runs)
    if args.augment and args.save_dir == 'weights/rec2':
        args.save_dir = 'weights/rec2_aug'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Enable TF32
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('TF32 enabled for faster training')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        vietocr_dir=args.vietocr_dir,
        val_annotation=args.val_annotation,
        val_img_dir=args.val_img_dir,
        test_annotation=args.test_annotation,
        test_img_dir=args.val_img_dir,  # same image dir for test
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width),
        num_workers=args.num_workers,
        augment=args.augment,
    )

    # Model
    model = SVTRv2(variant=args.variant, in_channels=3).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: SVTRv2-{args.variant} ({total_params:.2f}M params)')

    # Loss
    criterion = SVTRv2Loss(
        pad_id=model.tokenizer.pad_id,
        lambda_sgm=args.lambda_sgm,
        zero_infinity=True
    )

    # Optimizer (AdamW, following SVTRv2 paper)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda')

    # OneCycleLR scheduler (per-batch stepping)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.075,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000
    )

    # Resume from checkpoint (model weights only — fresh optimizer/scheduler)
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        prev_epoch = checkpoint.get('epoch', '?')
        prev_acc = checkpoint.get('best_acc', 0.0)
        print(f'Loaded weights from {args.resume}')
        print(f'  Previous training: epoch {prev_epoch}, accuracy {prev_acc:.4f}')
        print(f'  Starting fresh optimizer + LR schedule for continued training')

    # CSV log
    log_file = save_dir / 'training_log.csv'
    if not log_file.exists() or not args.resume:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_cer', 'val_accuracy'])

    print(f'\n{"="*80}')
    print(f'Starting SVTRv2-{args.variant} training...')
    print(f'{"="*80}\n')

    for epoch in range(args.epochs):
        # Train (with SGM)
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, scheduler, device, epoch + 1
        )

        # Validate (CTC only, no SGM)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

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

        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_metrics': val_metrics,
                'variant': args.variant
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f'  → Saved best model (Accuracy: {best_acc:.4f})')

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_metrics': val_metrics,
                'variant': args.variant
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')

    # ── Final test evaluation ──
    if test_loader:
        print(f'\n{"="*80}')
        print('Running test evaluation...')
        test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f} | '
              f'Test CER: {test_metrics["cer"]:.4f} | '
              f'Test Accuracy: {test_metrics["accuracy"]:.4f}')

    print(f'\n{"="*80}')
    print(f'Training completed! Best Val Accuracy: {best_acc:.4f}')
    print(f'Model saved to: {save_dir / "best_model.pth"}')
    print(f'Training log: {log_file}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
