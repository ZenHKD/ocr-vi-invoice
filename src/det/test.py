import os
import sys
import cv2
import torch
import numpy as np
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.det.dbnet import DBNetPP
from src.preprocess.scanner import preprocess_image

import pyclipper
from shapely.geometry import Polygon

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    if len(_box) == 0:
        return 0
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, [box.reshape(-1, 2).astype(int)], 1)
    return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


class DBPostProcessor:
    def __init__(self, thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.min_area = 10 # Default minimum area

    def __call__(self, pred, is_output_polygon=False):
        # pred: (C, H, W) -> we expect (1, H, W)
        segmentation = pred[0] > self.thresh

        # Find contours
        contours, _ = cv2.findContours((segmentation * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scores = []
        
        for i, contour in enumerate(contours):
            if i >= self.max_candidates:
                break
                
            epsilon = 0.002 * cv2.arcLength(contour, True) # Smaller epsilon for more detailed polygons
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            
            if points.shape[0] < 4:
                continue
                
            score = box_score_fast(pred[0], points)
            if self.box_thresh > score:
                continue
            
            # Filter by area
            if cv2.contourArea(points) < self.min_area:
                 continue
            
            if points.shape[0] > 2:
                # Unclip (expand)
                # Ensure the polygon is valid
                try:
                    expanded = unclip(points, unclip_ratio=self.unclip_ratio)
                    if len(expanded) > 0:
                        box = np.array(expanded[0])
                    else: 
                        continue
                except Exception as e:
                    # Fallback or skip
                    continue
            else:
                continue
            
            box = box.reshape(-1, 2)
            if len(box) < 4:
                continue
                
            boxes.append(box)
            scores.append(score)
            
        return boxes, scores


def resize_image(image, image_size=640):
    h, w = image.shape[:2]
    # Resize such that the larger side is image_size
    scale = image_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Ensure divisible by 32 (stride of ResNet50)
    new_h = int(np.round(new_h / 32) * 32)
    new_w = int(np.round(new_w / 32) * 32)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    
    return resized_image, (new_h / h, new_w / w)

def crop_image(img, box):
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(box)
    x = max(0, x)
    y = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    return img[y:y+bh, x:x+bw]


def load_model(model_path, device):
    model = DBNetPP(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Remove module. prefix
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


def main():
    parser = argparse.ArgumentParser(description='DBNet++ Inference')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a single image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--visualize_crops', action='store_true', help='Visualize cropped text regions')
    parser.add_argument('--image_size', type=int, default=960, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    # Post-processing args
    parser.add_argument('--thresh', type=float, default=0.3, help='Binarization threshold')
    parser.add_argument('--box_thresh', type=float, default=0.6, help='Box score threshold')
    parser.add_argument('--unclip_ratio', type=float, default=1.5, help='Unclip ratio (increase to merge fragmented text regions)')
    parser.add_argument('--min_area', type=float, default=10, help='Minimum area for text region')
    parser.add_argument('--preprocess', action='store_true', help='Use Document Scanner (U-2-Net) preprocessing')

    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    
    # Load model
    print(f'Loading model from {args.model_path}...')
    if not os.path.exists(args.model_path):
        print(f'Error: Model file {args.model_path} not found.')
        return

    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    post_processor = DBPostProcessor(
        thresh=args.thresh, 
        box_thresh=args.box_thresh, 
        unclip_ratio=args.unclip_ratio,
    )
    post_processor.min_area = args.min_area
    
    # Process images
    image_paths = []
    if args.image_path:
        p = Path(args.image_path)
        if p.exists():
            image_paths = [p]
        else:
            print(f"Error: Image {args.image_path} does not exist.")
            return
    
    print(f'Found {len(image_paths)} images to process')
    
    # ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for image_path in image_paths:
        print(f'Processing {image_path.name}...')
        
        # Load and preprocess
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            print(f'Failed to load {image_path}')
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Preprocessing (Document Scanner)
        if args.preprocess:
            print("Running AI Preprocessing...")
            try:
                # preprocess_image expects BGR image (OpenCV default)
                # But original_image is RGB here. Convert to BGR.
                img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                processed_bgr = preprocess_image(img_bgr, enhance=False)
                
                if processed_bgr is not None:
                     # Convert back to RGB for model
                     original_image = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                     print("Preprocessing complete.")
                else:
                     print("Preprocessing failed (no doc found), using original.")
            except Exception as e:
                print(f"Preprocessing error: {e}")
        
        # Resize
        resized_image, (scale_h, scale_w) = resize_image(original_image, args.image_size)
        
        # Normalize
        img_input = resized_image.astype(np.float32) / 255.0
        img_input = (img_input - mean) / std
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            preds = model(img_tensor) # Returns dict usually
            
            # Access 'binary' output
            if isinstance(preds, dict):
                pred_binary = preds['binary']
            else:
                pred_binary = preds
                
        # Post-processing
        prob_map = pred_binary[0].cpu().numpy() # (1, H, W)
        
        boxes, scores = post_processor(prob_map)
        
        # Rescale boxes to original image size
        rescaled_boxes = []
        for box in boxes:
            box[:, 0] = box[:, 0] / scale_w
            box[:, 1] = box[:, 1] / scale_h
            rescaled_boxes.append(box.astype(np.int32))
            
        elapsed = time.time() - start_time
        print(f'Found {len(rescaled_boxes)} text regions (Time: {elapsed:.4f}s)')
        
        # Visualization
        viz_image = original_image.copy()
        cv2.drawContours(viz_image, rescaled_boxes, -1, (0, 255, 0), 2)

        if args.visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(viz_image)
            plt.axis('off')
            plt.title(f'{image_path.name} - {len(rescaled_boxes)} detections')
            plt.show()

        if args.visualize_crops and len(rescaled_boxes) > 0:
            print(f"Visualizing {len(rescaled_boxes)} crops...")
            num_crops = len(rescaled_boxes)
            cols = 5
            rows = (num_crops + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            axes = axes.flatten() if num_crops > 1 else [axes]
            
            for i, box in enumerate(rescaled_boxes):
                crop = crop_image(original_image, box)
                if crop.size == 0:
                    print(f"Warning: Crop {i} is empty. Skipping.")
                    continue
                axes[i].imshow(crop)
                axes[i].axis('off')
                axes[i].set_title(f'Crop {i}')
            
            # Hide empty subplots
            for i in range(num_crops, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    main()
