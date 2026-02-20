"""
OCR Pipeline v2 — DBNet++ Detection + SVTRv2 Recognition

Usage:
  python src/pipeline/pipeline2.py \
      --det_model weights/det/best_model.pth \
      --rec_model weights/rec2/best_model.pth \
      --image_path data/test_images/ \
      --visualize --save_result
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.det.dbnet import DBNetPP
from model.rec2.svtrv2 import SVTRv2
from src.preprocess.scanner import preprocess_image
from src.det.test import crop_image, DBPostProcessor


# ── Detection helpers ──

def resize_image_for_det(image, image_size=640):
    """Resize image for detection model, ensuring divisible by 32."""
    h, w = image.shape[:2]
    scale = image_size / max(h, w)
    new_h = int(np.round(h * scale / 32) * 32)
    new_w = int(np.round(w * scale / 32) * 32)
    resized = cv2.resize(image, (new_w, new_h))
    return resized, (new_h / h, new_w / w)


def load_detection_model(model_path: str, device: str):
    """Load DBNet++ detection model."""
    model = DBNetPP(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Remove 'module.' prefix if present (DataParallel)
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Print checkpoint info
    epoch = checkpoint.get('epoch', '?')
    best_f1 = checkpoint.get('best_f1', '?')
    val_metrics = checkpoint.get('val_metrics', {})
    print(f'  Loaded epoch {epoch}, best F1: {best_f1}')
    if val_metrics:
        print(f'  Val metrics: P={val_metrics.get("precision", "?"):.4f} '
              f'R={val_metrics.get("recall", "?"):.4f} '
              f'F1={val_metrics.get("f1", "?"):.4f} '
              f'IoU={val_metrics.get("iou", "?"):.4f}')

    return model


# ── Recognition helpers ──

def load_recognition_model(model_path: str, device: str, variant: str = 'base'):
    """Load SVTRv2 recognition model."""
    model = SVTRv2(variant=variant, in_channels=3).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    variant_info = checkpoint.get('variant', variant)
    epoch = checkpoint.get('epoch', '?')
    best_acc = checkpoint.get('best_acc', '?')
    best_cer = checkpoint.get('val_metrics', {}).get('cer', '?')
    print(f"  SVTRv2-{variant_info} loaded (epoch {epoch}, best_acc={best_acc}, best_cer={best_cer})")
    return model


def preprocess_for_recognition(crop: np.ndarray,
                                img_size: Tuple[int, int] = (32, 256)) -> torch.Tensor:
    """
    Preprocess cropped text image for SVTRv2.
    Same logic as dataloader: resize height → pad width with white.
    """
    h, w = crop.shape[:2]
    target_h, target_w = img_size

    scale = target_h / h
    new_w = int(w * scale)

    if new_w > target_w:
        resized = cv2.resize(crop, (target_w, target_h))
    else:
        resized = cv2.resize(crop, (new_w, target_h))
        pad_w = target_w - new_w
        if pad_w > 0:
            if len(resized.shape) == 2:
                resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_w,
                                             cv2.BORDER_CONSTANT, value=255)
            else:
                resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_w,
                                             cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Convert to RGB
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)

    # Normalize (ImageNet)
    img_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor


def recognize_text(model: SVTRv2, crop: np.ndarray, device: str,
                   img_size: Tuple[int, int] = (32, 256)) -> str:
    """Run SVTRv2 recognition on a single cropped text image."""
    img_tensor = preprocess_for_recognition(crop, img_size=img_size)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(img_tensor)  # (T, 1, num_classes)
        predictions = model.decode_probs(log_probs)

    return predictions[0] if predictions else ""


def recognize_text_batch(model: SVTRv2, crops: List[np.ndarray], device: str,
                         img_size: Tuple[int, int] = (32, 256),
                         batch_size: int = 32) -> List[str]:
    """Batch recognition for better GPU utilization."""
    all_texts = []

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        tensors = []
        for crop in batch_crops:
            if crop.size == 0:
                # Create blank tensor for empty crops
                tensors.append(torch.zeros(3, img_size[0], img_size[1]))
            else:
                tensors.append(preprocess_for_recognition(crop, img_size))

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            log_probs = model(batch_tensor)
            predictions = model.decode_probs(log_probs)

        all_texts.extend(predictions)

    return all_texts


# ── Visualization ──

def draw_boxes_with_text(image: np.ndarray, boxes: List[np.ndarray],
                         texts: List[str], color=(0, 255, 0)) -> np.ndarray:
    """Draw detected boxes with numbered labels."""
    viz_img = image.copy()

    for idx, (box, text) in enumerate(zip(boxes, texts)):
        cv2.drawContours(viz_img, [box.astype(np.int32)], -1, color, 2)

        top_point = tuple(box[box[:, 1].argmin()])
        text_pos = (int(top_point[0]), int(top_point[1]) - 10)

        if text_pos[1] < 20:
            text_pos = (text_pos[0], int(box[:, 1].max()) + 20)

        cv2.putText(viz_img, str(idx + 1), text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return viz_img


# ── Main pipeline ──

def main():
    parser = argparse.ArgumentParser(description='OCR Pipeline v2 — DBNet++ + SVTRv2')

    # Model paths
    parser.add_argument('--det_model', type=str, required=True,
                        help='Path to DBNet++ detection checkpoint')
    parser.add_argument('--rec_model', type=str, required=True,
                        help='Path to SVTRv2 recognition checkpoint')
    parser.add_argument('--variant', type=str, default='base',
                        choices=['tiny', 'small', 'base'],
                        help='SVTRv2 variant')

    # Input
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to test image or directory')

    # Detection params
    parser.add_argument('--det_size', type=int, default=960)
    parser.add_argument('--det_thresh', type=float, default=0.3)
    parser.add_argument('--det_box_thresh', type=float, default=0.5)
    parser.add_argument('--det_unclip_ratio', type=float, default=1.6)
    parser.add_argument('--det_min_area', type=float, default=10)

    # Recognition params
    parser.add_argument('--rec_img_height', type=int, default=32)
    parser.add_argument('--rec_img_width', type=int, default=256)
    parser.add_argument('--rec_batch_size', type=int, default=32,
                        help='Batch size for recognition inference')

    # Preprocessing
    parser.add_argument('--preprocess', action='store_true',
                        help='Use Document Scanner preprocessing')

    # Visualization
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize_crops', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--output_dir', type=str, default='outputs')

    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    device = torch.device(args.device)
    print(f'Using device: {device}')

    if args.save_result:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──
    print(f'Loading detection model: {args.det_model}')
    det_model = load_detection_model(args.det_model, device)

    print(f'Loading recognition model: {args.rec_model}')
    rec_model = load_recognition_model(args.rec_model, device, variant=args.variant)

    # Post-processor
    post_processor = DBPostProcessor(
        thresh=args.det_thresh,
        box_thresh=args.det_box_thresh,
        unclip_ratio=args.det_unclip_ratio
    )
    post_processor.min_area = args.det_min_area

    # ── Collect images ──
    image_path = Path(args.image_path)
    if image_path.is_dir():
        image_paths = sorted(
            list(image_path.glob('*.jpg')) +
            list(image_path.glob('*.png')) +
            list(image_path.glob('*.jpeg'))
        )
    else:
        image_paths = [image_path]

    print(f'Found {len(image_paths)} images to process\n')

    # ImageNet stats for detection preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rec_size = (args.rec_img_height, args.rec_img_width)

    for img_path in image_paths:
        print(f'Processing: {img_path.name}')
        print('=' * 60)

        # Load image
        original_image = cv2.imread(str(img_path))
        if original_image is None:
            print(f'  Failed to load {img_path}')
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # === STEP 1: Preprocessing ===
        if args.preprocess:
            print("  [1/4] Running Document Scanner preprocessing...")
            try:
                img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                processed_bgr = preprocess_image(img_bgr, enhance=False)
                if processed_bgr is not None:
                    original_image = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                    print("        Done.")
                else:
                    print("        Failed, using original.")
            except Exception as e:
                print(f"        Error: {e}")
        else:
            print("  [1/4] Skipping preprocessing")

        # === STEP 2: Text Detection ===
        print("  [2/4] Running text detection...")
        resized_image, (scale_h, scale_w) = resize_image_for_det(
            original_image, args.det_size
        )

        img_input = resized_image.astype(np.float32) / 255.0
        img_input = (img_input - mean) / std
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            det_preds = det_model(img_tensor)
            pred_binary = det_preds['binary'] if isinstance(det_preds, dict) else det_preds

        prob_map = pred_binary[0].cpu().numpy()
        boxes, scores = post_processor(prob_map)

        # Rescale boxes to original image
        rescaled_boxes = []
        for box in boxes:
            box[:, 0] = box[:, 0] / scale_w
            box[:, 1] = box[:, 1] / scale_h
            rescaled_boxes.append(box.astype(np.int32))

        print(f"        Detected {len(rescaled_boxes)} text regions")

        if len(rescaled_boxes) == 0:
            print("        No text detected, skipping recognition.\n")
            continue

        # === STEP 3: Text Recognition (batched) ===
        print(f"  [3/4] Running SVTRv2 recognition ({len(rescaled_boxes)} crops)...")

        # Crop all regions
        crops = []
        for box in rescaled_boxes:
            crop = crop_image(original_image, box)
            crops.append(crop)

        # Batch inference
        recognized_texts = recognize_text_batch(
            rec_model, crops, device,
            img_size=rec_size,
            batch_size=args.rec_batch_size
        )

        for i, text in enumerate(recognized_texts):
            print(f"        Region {i+1}: '{text}'")

        # === STEP 4: Visualization & Output ===
        print("  [4/4] Generating outputs...")

        result_image = draw_boxes_with_text(
            original_image, rescaled_boxes, recognized_texts
        )

        if args.visualize:
            plt.figure(figsize=(8, 8), dpi=200)
            plt.imshow(result_image)
            plt.axis('off')
            plt.title(f'OCR Pipeline v2 — {img_path.name}\n'
                      f'Detected {len(rescaled_boxes)} regions (SVTRv2-{args.variant})')
            plt.tight_layout()
            plt.show()

        if args.visualize_crops and len(rescaled_boxes) > 0:
            print(f"\n  Displaying {len(rescaled_boxes)} crops...")
            num_crops = len(rescaled_boxes)
            cols = min(5, num_crops)
            rows = (num_crops + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if num_crops == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, (box, text) in enumerate(zip(rescaled_boxes, recognized_texts)):
                crop = crop_image(original_image, box)
                if crop.size == 0:
                    continue
                axes[i].imshow(crop)
                axes[i].set_title(f'{text}', fontsize=20, color='blue', fontweight='bold')
                axes[i].axis('off')

            for i in range(num_crops, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

        if args.save_result:
            output_path = output_dir / f'result_{img_path.stem}.jpg'
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), result_bgr)
            print(f"  Saved result to: {output_path}")

        print('\n' + '=' * 60 + '\n')

    print('Pipeline v2 completed!')


if __name__ == '__main__':
    main()
