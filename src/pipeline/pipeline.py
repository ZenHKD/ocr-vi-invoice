import os
import sys
import cv2
import torch
import numpy as np
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.det.dbnet import DBNetPP
from model.rec.svtr_ctc import SVTRCTC
from src.preprocess.scanner import preprocess_image
from src.det.test import box_score_fast, unclip, crop_image, DBPostProcessor

# Import post-processing from test.py
import pyclipper
from shapely.geometry import Polygon

def resize_image_for_det(image, image_size=640):
    """Resize image for detection model"""
    h, w = image.shape[:2]
    scale = image_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Ensure divisible by 32 (stride of ResNet50)
    new_h = int(np.round(new_h / 32) * 32)
    new_w = int(np.round(new_w / 32) * 32)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image, (new_h / h, new_w / w)


def load_detection_model(model_path: str, device: str):
    """Load DBNet++ detection model"""
    model = DBNetPP(backbone='resnet50', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Remove module. prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def load_recognition_model(model_path: str, device: str, img_size: Tuple[int, int] = (32, 384)):
    """Load SVTR-CTC recognition model"""
    model = SVTRCTC(img_size=img_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model


def preprocess_for_recognition(crop: np.ndarray, img_size: Tuple[int, int] = (32, 128)) -> torch.Tensor:
    """
    Preprocess cropped image for recognition model.
    Resizes to fixed height with variable width (maintaining aspect ratio).
    """
    h, w = crop.shape[:2]
    target_h, target_w = img_size  # target_w is max reference
    
    # Scale to fixed height
    scale = target_h / h
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Ensure width is divisible by 4 (network stride)
    if new_w % 4 != 0:
        new_w = ((new_w // 4) + 1) * 4
    
    # Resize
    resized = cv2.resize(crop, (new_w, new_h))
    
    # Convert to RGB if needed
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
    elif resized.shape[2] == 3:
        # Input is already RGB (from main loop)
        pass
    
    # Normalize
    img_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor


def recognize_text(model: SVTRCTC, crop: np.ndarray, device: str) -> str:
    """Run recognition on a single cropped text image"""
    # Preprocess
    img_tensor = preprocess_for_recognition(crop)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        log_probs = model(img_tensor)  # (T, 1, num_classes)
        predictions = model.decode_probs(log_probs)
    
    return predictions[0] if predictions else ""


def draw_boxes_with_text(image: np.ndarray, boxes: List[np.ndarray], 
                         texts: List[str], color=(0, 255, 0)) -> np.ndarray:
    viz_img = image.copy()
    
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        cv2.drawContours(viz_img, [box.astype(np.int32)], -1, color, 2)
        
        top_point = tuple(box[box[:, 1].argmin()])
        text_pos = (int(top_point[0]), int(top_point[1]) - 10)
        
        if text_pos[1] < 20:
            text_pos = (text_pos[0], int(box[:, 1].max()) + 20)
        
        # Red text with fixed size
        cv2.putText(viz_img, str(idx + 1), text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return viz_img

def main():
    parser = argparse.ArgumentParser(description='Full OCR Pipeline - Detection + Recognition')
    
    # Model paths
    parser.add_argument('--det_model', type=str, required=True, 
                       help='Path to DBNet++ detection model checkpoint')
    parser.add_argument('--rec_model', type=str, required=True,
                       help='Path to SVTR-CTC recognition model checkpoint')
    
    # Input
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to test image or directory of images')
    
    # Detection parameters
    parser.add_argument('--det_size', type=int, default=960,
                       help='Input size for detection model')
    parser.add_argument('--det_thresh', type=float, default=0.8,
                       help='Binarization threshold for detection')
    parser.add_argument('--det_box_thresh', type=float, default=0.8,
                       help='Box score threshold for detection')
    parser.add_argument('--det_unclip_ratio', type=float, default=0.8,
                       help='Unclip ratio for detection post-processing')
    parser.add_argument('--det_min_area', type=float, default=10,
                       help='Minimum area for text region')
    
    # Recognition parameters
    parser.add_argument('--rec_img_height', type=int, default=32,
                       help='Image height for recognition model')
    parser.add_argument('--rec_img_width', type=int, default=384,
                       help='Max image width for recognition model')
    
    # Preprocessing
    parser.add_argument('--preprocess', action='store_true',
                       help='Use AI Document Scanner preprocessing')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize final result with detected boxes and text')
    parser.add_argument('--visualize_crops', action='store_true',
                       help='Visualize each cropped region with predicted text as title')
    parser.add_argument('--save_result', action='store_true',
                       help='Save result image to output directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save output images')
    
    # Device
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create output directory
    if args.save_result:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print(f'Loading detection model from {args.det_model}...')
    det_model = load_detection_model(args.det_model, device)
    
    print(f'Loading recognition model from {args.rec_model}...')
    rec_model = load_recognition_model(
        args.rec_model, 
        device, 
        img_size=(args.rec_img_height, args.rec_img_width)
    )
    
    # Initialize post-processor
    post_processor = DBPostProcessor(
        thresh=args.det_thresh,
        box_thresh=args.det_box_thresh,
        unclip_ratio=args.det_unclip_ratio
    )
    post_processor.min_area = args.det_min_area
    
    # Collect images to process
    image_path = Path(args.image_path)
    if image_path.is_dir():
        image_paths = list(image_path.glob('*.jpg')) + \
                     list(image_path.glob('*.png')) + \
                     list(image_path.glob('*.jpeg'))
    else:
        image_paths = [image_path]
    
    print(f'Found {len(image_paths)} images to process\n')
    
    # ImageNet normalization for detection
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Process each image
    for img_path in image_paths:
        print(f'Processing: {img_path.name}')
        print('=' * 60)
        
        # Load image
        original_image = cv2.imread(str(img_path))
        if original_image is None:
            print(f'Failed to load {img_path}')
            continue
        
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        
        # === STEP 1: Preprocessing (Optional) ===
        if args.preprocess:
            print("  [1/4] Running U^2-Net Document Scanner preprocessing...")
            try:
                img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                processed_bgr = preprocess_image(img_bgr, enhance=False)
                if processed_bgr is not None:
                    original_image = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                    print("        Preprocessing complete.")
                else:
                    print("        Preprocessing failed, using original.")
            except Exception as e:
                print(f"        Preprocessing error: {e}")
        else:
            print("  [1/4] Skipping preprocessing")
        
        # === STEP 2: Text Detection ===
        print("  [2/4] Running text detection...")
        resized_image, (scale_h, scale_w) = resize_image_for_det(
            original_image, args.det_size
        )
        
        # Normalize for detection
        img_input = resized_image.astype(np.float32) / 255.0
        img_input = (img_input - mean) / std
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            det_preds = det_model(img_tensor)
            if isinstance(det_preds, dict):
                pred_binary = det_preds['binary']
            else:
                pred_binary = det_preds
        
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
        
        # === STEP 3: Text Recognition ===
        print("  [3/4] Running text recognition on crops...")
        recognized_texts = []
        
        for i, box in enumerate(rescaled_boxes):
            # Crop text region
            crop = crop_image(original_image, box)
            if crop.size == 0:
                recognized_texts.append("")
                continue
            
            # Recognize text
            text = recognize_text(rec_model, crop, device)
            recognized_texts.append(text)
            print(f"        Region {i+1}: '{text}'")
        
        # === STEP 4: Visualization & Output ===
        print("  [4/4] Generating outputs...")
        
        # Create visualization with boxes and text
        result_image = draw_boxes_with_text(
            original_image, rescaled_boxes, recognized_texts
        )
        
        elapsed = time.time() - start_time
        print(f"\n  Total time: {elapsed:.3f}s")
        
        # Visualize final result (if requested)
        if args.visualize:
            plt.figure(figsize=(8, 8), dpi=200)
            plt.imshow(result_image)
            plt.axis('off')
            plt.title(f'OCR Pipeline Result - {img_path.name}\nDetected {len(rescaled_boxes)} regions')
            plt.tight_layout()
            plt.show()

        # Visualize crops with text as title (if requested)
        if args.visualize_crops and len(rescaled_boxes) > 0:
            print(f"\n  Displaying {len(rescaled_boxes)} cropped regions...")
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
                
                # Convert to RGB for display if needed
                if len(crop.shape) == 3 and crop.shape[2] == 3:
                    display_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                else:
                    display_crop = crop
                
                axes[i].imshow(display_crop)
                axes[i].set_title(f'{text}', fontsize=20, color='blue', fontweight='bold')
                axes[i].axis('off')
            
            # Hide empty subplots
            for i in range(num_crops, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Save result (if requested)
        if args.save_result:
            output_path = output_dir / f'result_{img_path.stem}.jpg'
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), result_bgr)
            print(f"  Saved result to: {output_path}")
        
        print('\n' + '=' * 60 + '\n')
    
    print('Pipeline completed!')


if __name__ == '__main__':
    main()
