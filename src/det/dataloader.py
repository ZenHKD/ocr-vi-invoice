"""
DBNet++ DataLoader for Text Detection
Loads invoice images with polygon annotations and generates ground truth maps.

Features:
    - Polygon shrinkage via Vatti Clipping (core DBNet technique)
    - FAST threshold map via cv2.distanceTransform (vectorized, no Python loops)
    - Albumentations for fast, optimized augmentation
"""

import os
import json
import cv2
import random
import numpy as np
import pyclipper
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon as ShapelyPolygon

import albumentations as A


class DetectionDataset(Dataset):
    """Dataset for text detection with DBNet++"""
    
    def __init__(self, data_dir: str, image_size: int = 640, 
                 is_training: bool = False, shrink_ratio: float = 0.4,
                 thresh_min: float = 0.3, thresh_max: float = 0.7):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        
        # Find all JSON annotation files
        self.samples = sorted(list(self.data_dir.glob("*.json")))
        
        # Normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Albumentations augmentation pipeline (much faster than manual cv2)
        if is_training:
            self.augment = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.02, scale_limit=0.3, rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
                ),
                A.HorizontalFlip(p=0.1),
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5
                ),
            ], keypoint_params=A.KeypointParams(
                format='xy', remove_invisible=False
            ))
        else:
            self.augment = None

    def __len__(self):
        return len(self.samples)
    
    # ========================================================================
    # Polygon Shrinkage (Vatti Clipping)
    # ========================================================================
    
    def _shrink_polygon(self, polygon: np.ndarray) -> Optional[np.ndarray]:
        """Shrink polygon using Vatti clipping. D = A * (1 - r^2) / L"""
        try:
            poly = ShapelyPolygon(polygon)
            if not poly.is_valid or poly.area < 1:
                return None
            
            area = poly.area
            perimeter = poly.length
            if perimeter < 1:
                return None
            
            d = area * (1 - self.shrink_ratio ** 2) / perimeter
            
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(
                [tuple(p) for p in polygon.astype(int)],
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON
            )
            
            shrunk = pco.Execute(-d)
            if not shrunk:
                return None
            
            shrunk = max(shrunk, key=lambda x: pyclipper.Area(x))
            shrunk = np.array(shrunk, dtype=np.float32)
            
            return shrunk if len(shrunk) >= 3 else None
            
        except Exception:
            return None
    
    def _dilate_polygon(self, polygon: np.ndarray) -> Optional[np.ndarray]:
        """Dilate polygon for threshold map border."""
        try:
            poly = ShapelyPolygon(polygon)
            if not poly.is_valid or poly.area < 1:
                return None
            
            area = poly.area
            perimeter = poly.length
            if perimeter < 1:
                return None
            
            d = area * (1 - self.shrink_ratio ** 2) / perimeter
            
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(
                [tuple(p) for p in polygon.astype(int)],
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON
            )
            
            dilated = pco.Execute(d)
            if not dilated:
                return None
            
            dilated = max(dilated, key=lambda x: pyclipper.Area(x))
            return np.array(dilated, dtype=np.float32)
            
        except Exception:
            return None
    
    # ========================================================================
    # FAST Threshold Map (vectorized with cv2.distanceTransform)
    # ========================================================================
    
    def _draw_border_map(self, polygon: np.ndarray,
                          thresh_map: np.ndarray, thresh_mask: np.ndarray):
        """
        Generate threshold map using cv2.distanceTransform (C++ vectorized).
        ~100x faster than the pixel-level Python loop.
        """
        dilated = self._dilate_polygon(polygon)
        if dilated is None:
            return
        
        h, w = thresh_map.shape
        
        # Compute shrink distance
        poly = ShapelyPolygon(polygon)
        if not poly.is_valid:
            return
        area = poly.area
        perimeter = poly.length
        if perimeter < 1:
            return
        d = area * (1 - self.shrink_ratio ** 2) / perimeter
        if d < 1:
            return
        
        # Mark border region in thresh_mask
        cv2.fillPoly(thresh_mask, [dilated.astype(np.int32)], 1.0)
        
        # Create binary mask of polygon interior
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], 1)
        
        # Distance transform: distance from each pixel to nearest polygon edge
        # For pixels INSIDE the polygon: distance to boundary
        dist_inside = cv2.distanceTransform(poly_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # For pixels OUTSIDE the polygon: distance to boundary
        dist_outside = cv2.distanceTransform(1 - poly_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # Combined distance to boundary (min of inside/outside distance)
        dist = np.minimum(dist_inside, dist_outside)
        
        # Create dilated region mask
        dilated_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(dilated_mask, [dilated.astype(np.int32)], 1)
        
        # Only compute in the border region (dilated area)
        border_region = dilated_mask.astype(bool)
        
        # Normalize to [0, 1] clamped by d
        norm_dist = np.clip(dist / d, 0, 1.0)
        
        # Map to threshold range: closer to boundary = higher threshold
        thresh_vals = self.thresh_max - norm_dist * (self.thresh_max - self.thresh_min)
        
        # Apply only in border region, taking max with existing values
        update_mask = border_region & (thresh_vals > thresh_map)
        thresh_map[update_mask] = thresh_vals[update_mask]
    
    # ========================================================================
    # Augmentation (Albumentations)
    # ========================================================================
    
    def _apply_augmentation(self, image: np.ndarray, polygons: List[np.ndarray]):
        """Apply albumentations augmentation to image and polygons."""
        if self.augment is None or not polygons:
            return image, polygons
        
        # Flatten all polygon keypoints with polygon index tracking
        keypoints = []
        poly_indices = []
        point_indices = []
        for poly_idx, poly in enumerate(polygons):
            for pt_idx, pt in enumerate(poly):
                keypoints.append(tuple(pt))
                poly_indices.append(poly_idx)
                point_indices.append(pt_idx)
        
        try:
            result = self.augment(image=image, keypoints=keypoints)
            aug_image = result['image']
            aug_keypoints = result['keypoints']
            
            # Reconstruct polygons from augmented keypoints
            new_polygons = [[] for _ in range(len(polygons))]
            for kp, pi, pti in zip(aug_keypoints, poly_indices, point_indices):
                new_polygons[pi].append(kp)
            
            # Convert back to numpy arrays, filter invalid
            final_polygons = []
            for poly_pts in new_polygons:
                if len(poly_pts) >= 3:
                    final_polygons.append(np.array(poly_pts, dtype=np.float32))
            
            return aug_image, final_polygons if final_polygons else polygons
            
        except Exception:
            return image, polygons
    
    # ========================================================================
    # Resize + Pad
    # ========================================================================
    
    def _resize_pad(self, image: np.ndarray, masks: List[np.ndarray]):
        """Resize keeping aspect ratio, pad to target size, return tensors."""
        h, w = image.shape[:2]
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        if scale != 1.0:
            image = cv2.resize(image, (new_w, new_h))
        
        # Normalize image (in numpy, before tensor conversion)
        img_f = image.astype(np.float32) / 255.0
        img_f = (img_f - self.mean) / self.std
        
        # HWC -> CHW
        img_tensor = torch.from_numpy(img_f.transpose(2, 0, 1))
        
        # Pad image
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), value=0)
        
        # Resize and pad masks
        final_masks = []
        for mask in masks:
            if scale != 1.0:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask).float()
            if pad_h > 0 or pad_w > 0:
                mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=0)
            final_masks.append(mask_tensor.unsqueeze(0))
        
        return img_tensor, final_masks

    # ========================================================================
    # Main __getitem__
    # ========================================================================
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load image and generate ground truth maps with polygon shrinkage."""
        try:
            return self._load_sample(idx)
        except Exception as e:
            # Handle corrupt images / broken annotations gracefully
            print(f"Warning: Failed to load sample {idx}: {e}. Returning blank.")
            return self._blank_sample()
    
    def _blank_sample(self) -> Dict[str, torch.Tensor]:
        """Return a blank sample (used as fallback for corrupt data)."""
        s = self.image_size
        return {
            'image': torch.zeros(3, s, s),
            'gt': torch.zeros(1, s, s),
            'mask': torch.zeros(1, s, s),      # mask=0 → ignored in loss
            'thresh_map': torch.zeros(1, s, s),
            'thresh_mask': torch.zeros(1, s, s),
        }
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        json_path = self.samples[idx]
        
        # Load annotation
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # Load image
        image_path = self.data_dir / json_path.name.replace('.json', '.jpg')
        image = cv2.imread(str(image_path))
        if image is None:
            image_path = self.data_dir / json_path.name.replace('.json', '.png')
            image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image for {json_path.name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract polygons
        polygons = []
        for ann in annotation.get('annotations', []):
            polygon = np.array(ann['polygon'], dtype=np.float32)
            if len(polygon) >= 3:
                polygons.append(polygon)
        
        # Apply augmentation (training only, via albumentations)
        if self.is_training and polygons:
            image, polygons = self._apply_augmentation(image, polygons)
        
        # Generate ground truth maps
        h, w = image.shape[:2]
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        thresh_mask = np.zeros((h, w), dtype=np.float32)
        
        for polygon in polygons:
            # Clip polygon to image bounds
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
            
            # === Probability GT Map: use SHRUNK polygon ===
            shrunk = self._shrink_polygon(polygon)
            if shrunk is not None:
                cv2.fillPoly(gt, [shrunk.astype(np.int32)], 1.0)
            else:
                cv2.fillPoly(mask, [polygon.astype(np.int32)], 0.0)
            
            # === Threshold Map: only compute during training ===
            # Skip for validation/test — metrics only need binary vs GT,
            # and this is the main bottleneck (~120 pyclipper ops per image)
            if self.is_training:
                self._draw_border_map(polygon, thresh_map, thresh_mask)
        
        # Apply resize and pad
        image_tensor, mask_tensors = self._resize_pad(image, [gt, mask, thresh_map, thresh_mask])
        gt, mask, thresh_map, thresh_mask = mask_tensors
        
        return {
            'image': image_tensor,
            'gt': gt,
            'mask': mask,
            'thresh_map': thresh_map,
            'thresh_mask': thresh_mask
        }


def create_dataloaders(
    train_dir: str = 'data/train_det',
    val_dir: str = 'data/val_det_sroie',
    test_dir: Optional[str] = 'data/test_det_sroie',
    batch_size: int = 8,
    image_size: int = 640,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create training, validation, and optional test dataloaders.

    Training: Synthetic generated data (train_det)
    Validation: SROIE train split (real receipts)
    Test: SROIE test split (real receipts)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("Loading datasets...")

    # ── Training data (synthetic) — with augmentation ──
    train_dataset = DetectionDataset(
        train_dir, image_size=image_size, is_training=True
    )

    # ── Validation data (SROIE train) — NO augmentation ──
    val_dataset = DetectionDataset(
        val_dir, image_size=image_size, is_training=False
    )

    # ── Test data (SROIE test) — NO augmentation ──
    test_loader = None
    if test_dir and os.path.isdir(test_dir):
        test_dataset = DetectionDataset(
            test_dir, image_size=image_size, is_training=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    print(f"\nDataset summary:")
    print(f"  Train:      {len(train_dataset):>8,} samples (synthetic)")
    print(f"  Validation: {len(val_dataset):>8,} samples (SROIE)")
    if test_loader:
        print(f"  Test:       {len(test_dataset):>8,} samples (SROIE)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader

