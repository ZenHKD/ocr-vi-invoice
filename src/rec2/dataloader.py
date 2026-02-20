"""
SVTRv2 DataLoader for Text Recognition

Supports three data source formats:
  1. VietOCR paired-file format: folder with N.jpg + N.txt pairs (multiple subfolders)
  2. MCOCR tab-separated annotation: annotation.txt with 'filename  label' lines
  3. CSV format: labels.csv with columns 'filename', 'text'

Images are resized with aspect-ratio preservation and right-padded to fixed size.
Includes heavy augmentation pipeline using Albumentations for training.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

import albumentations as A

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.rec2.tokenizer import Tokenizer
from model.rec2.vocab import VOCAB


def get_train_augmentation():
    """
    Heavy augmentation pipeline for text recognition training.
    Simulates real-world degradations: blur, noise, lighting, distortion.
    Applied BEFORE resize_pad.
    """
    return A.Compose([
        # ── Geometric distortions ──
        A.OneOf([
            A.Affine(
                rotate=(-5, 5),
                shear=(-10, 10),
                scale=(0.9, 1.1),
                mode=cv2.BORDER_CONSTANT,
                cval=255,
                p=1.0,
            ),
            A.Perspective(scale=(0.02, 0.06), pad_mode=cv2.BORDER_CONSTANT,
                          pad_val=255, p=1.0),
        ], p=0.5),

        # ── Blur (camera shake, focus issues) ──
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),

        # ── Noise (sensor noise, compression artifacts) ──
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            A.ImageCompression(quality_lower=50, quality_upper=90, p=1.0),
        ], p=0.4),

        # ── Color / Lighting changes ──
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.5),

        # ── Shadows / Occlusion simulation ──
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
                           num_shadows_upper=2, shadow_dimension=5, p=1.0),
            A.CoarseDropout(max_holes=5, max_height=8, max_width=8,
                            fill_value=0, p=1.0),
        ], p=0.2),

        # ── Grayscale / channel manipulation ──
        A.ToGray(p=0.1),

        # ── Sharpening (scanned documents) ──
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.15),

        # ── Downscale+upscale (simulate low resolution) ──
        A.Downscale(scale_min=0.5, scale_max=0.8,
                    interpolation=cv2.INTER_LINEAR, p=0.15),
    ])


class RecognitionDataset(Dataset):
    """
    Base dataset for text recognition with SVTRv2.
    Handles a list of (image_path, text_label) samples.
    """

    def __init__(self, samples: List[Tuple[str, str]],
                 img_size: Tuple[int, int] = (32, 256),
                 augment: bool = False):
        self.samples = samples
        self.img_size = img_size
        self.tokenizer = Tokenizer(VOCAB)
        self.augment = augment
        self.transform = get_train_augmentation() if augment else None

        # Normalization (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def resize_pad(self, image):
        """Resize to fixed size with aspect-ratio preservation + right padding."""
        h, w = image.shape[:2]
        target_h, target_w = self.img_size

        scale = target_h / h
        new_w = int(w * scale)

        if new_w > target_w:
            image = cv2.resize(image, (target_w, target_h))
        else:
            image = cv2.resize(image, (new_w, target_h))
            pad_w = target_w - new_w
            if pad_w > 0:
                image = cv2.copyMakeBorder(
                    image, 0, 0, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )

        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        return img_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, text = self.samples[idx]

        try:
            np_img = np.fromfile(str(img_path), np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError
        except Exception:
            image = np.ones((32, 256, 3), dtype=np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation BEFORE resize_pad (on original-resolution image)
        if self.augment and self.transform is not None:
            image = self.transform(image=image)['image']

        target = self.tokenizer.encode([text])[0]
        target_length = len(target)

        image_tensor = self.resize_pad(image)

        return {
            'image': image_tensor,
            'target': target,
            'target_length': target_length,
            'text': text
        }


def load_vietocr_samples(data_dir: str, subfolders: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Load samples from VietOCR paired-file format.
    Each image N.jpg/N.png has a paired N.txt containing the label.
    """
    data_dir = Path(data_dir)
    samples = []

    if subfolders is None:
        subfolders = [d.name for d in data_dir.iterdir() if d.is_dir()]

    for folder in subfolders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(f"Warning: subfolder {folder} not found, skipping.")
            continue

        img_extensions = {'.jpg', '.jpeg', '.png'}
        img_files = [f for f in folder_path.iterdir()
                     if f.suffix.lower() in img_extensions]

        for img_file in img_files:
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                try:
                    label = txt_file.read_text(encoding='utf-8').strip()
                    if label:
                        samples.append((str(img_file), label))
                except Exception:
                    continue

    print(f"  VietOCR [{data_dir.name}]: loaded {len(samples)} samples from {len(subfolders)} folders")
    return samples


def load_mcocr_samples(annotation_file: str, img_dir: str) -> List[Tuple[str, str]]:
    """
    Load samples from MCOCR tab-separated annotation file.
    Format: 'filename  label' per line (tab or multi-space separated).
    """
    samples = []
    img_dir = Path(img_dir)

    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                filename, label = parts
                img_path = img_dir / filename
                if img_path.exists() and label.strip():
                    samples.append((str(img_path), label.strip()))

    print(f"  MCOCR [{Path(annotation_file).name}]: loaded {len(samples)} samples")
    return samples


def load_csv_samples(data_dir: str, labels_file: str = 'labels.csv') -> List[Tuple[str, str]]:
    """Load samples from CSV format (filename,text columns)."""
    data_dir = Path(data_dir)
    labels_path = data_dir / labels_file
    df = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    samples = []
    for _, row in df.iterrows():
        img_path = data_dir / row['filename']
        if row['text']:
            samples.append((str(img_path), row['text']))
    print(f"  CSV [{labels_file}]: loaded {len(samples)} samples")
    return samples


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for fixed-size images."""
    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])

    batch_size = images.shape[0]
    fixed_width = images.shape[3]
    input_length_val = fixed_width // 4  # SVTRv2 ConvStem downsamples width by 4
    input_lengths = torch.full((batch_size,), input_length_val, dtype=torch.long)

    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    targets_raw = [item['target'] for item in batch]
    targets = pad_sequence(targets_raw, batch_first=True, padding_value=1)

    return {
        'image': images,
        'target': targets,
        'target_length': target_lengths,
        'input_length': input_lengths,
        'text': texts
    }


def create_dataloaders(
    # VietOCR training data
    vietocr_dir: str = 'data/vietocr',
    vietocr_subfolders: Optional[List[str]] = None,
    # MCOCR validation data
    val_annotation: str = 'data/archive/text_recognition_train_data.txt',
    val_img_dir: str = 'data/archive/text_recognition_mcocr_data/text_recognition_mcocr_data',
    # MCOCR test data
    test_annotation: Optional[str] = 'data/archive/text_recognition_val_data.txt',
    test_img_dir: Optional[str] = None,
    # Common
    batch_size: int = 64,
    img_size: Tuple[int, int] = (32, 256),
    num_workers: int = 4,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create training, validation, and optional test dataloaders.

    Training: VietOCR paired-file data (all ~601K samples)
    Validation: MCOCR annotation file (text_recognition_train_data.txt, 5284 samples)
    Test: MCOCR annotation file (text_recognition_val_data.txt, 1299 samples)

    Args:
        augment: If True, apply heavy data augmentation to training set only.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if test_img_dir is None:
        test_img_dir = val_img_dir

    print("Loading datasets...")
    if augment:
        print("  ⚡ Heavy augmentation ENABLED for training")

    # ── Training data (VietOCR) — with optional augmentation ──
    train_samples = load_vietocr_samples(vietocr_dir, vietocr_subfolders)
    train_dataset = RecognitionDataset(train_samples, img_size=img_size, augment=augment)

    # ── Validation data (MCOCR) — NO augmentation ──
    val_samples = load_mcocr_samples(val_annotation, val_img_dir)
    val_dataset = RecognitionDataset(val_samples, img_size=img_size, augment=False)

    # ── Test data (MCOCR) — NO augmentation ──
    test_loader = None
    if test_annotation:
        test_samples = load_mcocr_samples(test_annotation, test_img_dir)
        test_dataset = RecognitionDataset(test_samples, img_size=img_size, augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    print(f"\nDataset summary:")
    print(f"  Train:      {len(train_dataset):>8,} samples (VietOCR{' + augment' if augment else ''})")
    print(f"  Validation: {len(val_dataset):>8,} samples (MCOCR)")
    if test_loader:
        print(f"  Test:       {len(test_dataset):>8,} samples (MCOCR)")

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

    return train_loader, val_loader, test_loader
