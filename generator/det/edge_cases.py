"""
Edge Cases Generator for unusual/difficult invoice scenarios.

Provides:
    - Partial scans (cropped, cut-off)
    - Multi-receipt composites (multiple receipts on one scan)
    - Extreme rotations and angles
    - Blank pages with artifacts
    - Unreadable/corrupted images
    - Overlapping documents
    - Mixed orientation receipts
    - Receipt on textured backgrounds
    - Photos of receipts (not scans)
"""

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Dict, Tuple
from .geometry import rotate_point


class EdgeCaseGenerator:
    """Generate edge case scenarios for robust model training."""

    @staticmethod
    def create_partial_scan(img: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """
        Crop image to simulate partial scan.
        
        Returns:
            Modified image and filtered annotations (only visible text boxes)
        """
        width, height = img.size
        
        # Random crop amount (remove 10-40% from one or two edges)
        crop_top = random.randint(0, int(height * 0.3)) if random.random() < 0.5 else 0
        crop_bottom = random.randint(0, int(height * 0.3)) if random.random() < 0.5 else 0
        crop_left = random.randint(0, int(width * 0.3)) if random.random() < 0.5 else 0
        crop_right = random.randint(0, int(width * 0.3)) if random.random() < 0.5 else 0
        
        # Ensure at least 40% of image remains
        new_width = width - crop_left - crop_right
        new_height = height - crop_top - crop_bottom
        if new_width < width * 0.4:
            crop_right = 0
        if new_height < height * 0.4:
            crop_bottom = 0
        
        # Crop image
        cropped = img.crop((crop_left, crop_top, width - crop_right, height - crop_bottom))
        
        # Filter and adjust annotations
        new_annotations = []
        for ann in annotations:
            polygon = ann['polygon']
            # Check if polygon is at least partially visible
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            
            if (max(x_coords) > crop_left and min(x_coords) < width - crop_right and
                max(y_coords) > crop_top and min(y_coords) < height - crop_bottom):
                # Adjust polygon coordinates
                new_polygon = [
                    [max(0, min(x - crop_left, new_width)), 
                     max(0, min(y - crop_top, new_height))]
                    for x, y in polygon
                ]
                new_annotations.append({
                    'text': ann['text'],
                    'polygon': new_polygon
                })
        
        return cropped, new_annotations

    @staticmethod
    def create_extreme_rotation(img: Image.Image, annotations: List[Dict], 
                               angle: float = None) -> Tuple[Image.Image, List[Dict]]:
        """
        Apply extreme rotation (15-45 degrees).
        
        Returns:
            Rotated image and adjusted annotations
        """
        if angle is None:
            # Random extreme angle
            angle = random.choice([
                random.uniform(15, 45),
                random.uniform(-45, -15),
                random.uniform(135, 180),
                random.uniform(-180, -135)
            ])
        
        # Rotate image with expand to fit
        rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        # Calculate rotation center (original image center)
        orig_width, orig_height = img.size
        center = (orig_width / 2, orig_height / 2)
        
        # New image dimensions
        new_width, new_height = rotated.size
        
        # Calculate offset due to expansion
        angle_rad = np.radians(angle)
        offset_x = (new_width - orig_width) / 2
        offset_y = (new_height - orig_height) / 2
        
        # Rotate annotations
        new_annotations = []
        for ann in annotations:
            polygon = ann['polygon']
            new_polygon = []
            for x, y in polygon:
                # Rotate point around center
                x_rot, y_rot = rotate_point((x, y), center, angle)
                # Add offset for expanded canvas
                x_rot += offset_x
                y_rot += offset_y
                new_polygon.append([int(x_rot), int(y_rot)])
            
            new_annotations.append({
                'text': ann['text'],
                'polygon': new_polygon
            })
        
        return rotated, new_annotations

    @staticmethod
    def create_multi_receipt_composite(imgs_with_annotations: List[Tuple[Image.Image, List[Dict]]],
                                      num_receipts: int = 2) -> Tuple[Image.Image, List[Dict]]:
        """
        Combine multiple receipts into one image (side by side or overlapping).
        
        Args:
            imgs_with_annotations: List of (image, annotations) tuples
            num_receipts: Number of receipts to combine (2-3)
            
        Returns:
            Composite image and combined annotations
        """
        num_receipts = min(num_receipts, len(imgs_with_annotations))
        selected = random.sample(imgs_with_annotations, num_receipts)
        
        if num_receipts == 2:
            # Side by side
            img1, ann1 = selected[0]
            img2, ann2 = selected[1]
            
            # Resize to similar heights
            target_height = min(img1.height, img2.height)
            img1 = img1.resize((int(img1.width * target_height / img1.height), target_height))
            img2 = img2.resize((int(img2.width * target_height / img2.height), target_height))
            
            # Create composite
            gap = random.randint(20, 50)
            composite_width = img1.width + gap + img2.width
            composite = Image.new('RGB', (composite_width, target_height), (255, 255, 255))
            composite.paste(img1, (0, 0))
            composite.paste(img2, (img1.width + gap, 0))
            
            # Adjust annotations for img2
            combined_annotations = ann1.copy()
            for ann in ann2:
                new_polygon = [[x + img1.width + gap, y] for x, y in ann['polygon']]
                combined_annotations.append({
                    'text': ann['text'],
                    'polygon': new_polygon
                })
            
            return composite, combined_annotations
        else:
            # Simple stacking for 3+ receipts
            total_height = sum(img.height for img, _ in selected) + (num_receipts - 1) * 30
            max_width = max(img.width for img, _ in selected)
            
            composite = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            combined_annotations = []
            y_offset = 0
            
            for img, ann in selected:
                composite.paste(img, (0, y_offset))
                
                # Adjust annotations
                for a in ann:
                    new_polygon = [[x, y + y_offset] for x, y in a['polygon']]
                    combined_annotations.append({
                        'text': a['text'],
                        'polygon': new_polygon
                    })
                
                y_offset += img.height + 30
            
            return composite, combined_annotations

    @staticmethod
    def create_textured_background(img: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """
        Place receipt on textured background (desk, table, etc).
        
        Returns:
            Image on textured background, annotations unchanged
        """
        width, height = img.size
        
        # Create larger canvas with texture
        bg_width = width + random.randint(100, 300)
        bg_height = height + random.randint(100, 300)
        
        # Generate texture (wood grain or fabric pattern)
        texture_type = random.choice(['wood', 'fabric', 'concrete'])
        
        if texture_type == 'wood':
            base_color = random.randint(150, 200)
            bg = Image.new('RGB', (bg_width, bg_height), (base_color, base_color - 20, base_color - 40))
            # Add grain noise
            bg_array = np.array(bg)
            noise = np.random.randint(-20, 20, (bg_height, bg_width, 3))
            bg_array = np.clip(bg_array + noise, 0, 255).astype(np.uint8)
            bg = Image.fromarray(bg_array)
        elif texture_type == 'fabric':
            base_color = random.randint(100, 180)
            bg = Image.new('RGB', (bg_width, bg_height), (base_color, base_color, base_color + 10))
            # Add fabric texture
            bg_array = np.array(bg)
            noise = np.random.randint(-15, 15, (bg_height, bg_width, 3))
            bg_array = np.clip(bg_array + noise, 0, 255).astype(np.uint8)
            bg = Image.fromarray(bg_array)
        else:  # concrete
            base_color = random.randint(120, 160)
            bg = Image.new('RGB', (bg_width, bg_height), (base_color, base_color, base_color))
            bg_array = np.array(bg)
            noise = np.random.randint(-25, 25, (bg_height, bg_width, 3))
            bg_array = np.clip(bg_array + noise, 0, 255).astype(np.uint8)
            bg = Image.fromarray(bg_array)
        
        # Paste receipt at random position
        x_offset = random.randint(50, bg_width - width - 50)
        y_offset = random.randint(50, bg_height - height - 50)
        bg.paste(img, (x_offset, y_offset))
        
        # Adjust annotations
        new_annotations = []
        for ann in annotations:
            new_polygon = [[x + x_offset, y + y_offset] for x, y in ann['polygon']]
            new_annotations.append({
                'text': ann['text'],
                'polygon': new_polygon
            })
        
        return bg, new_annotations

    @staticmethod
    def create_blank_with_artifacts(size: Tuple[int, int] = (600, 800)) -> Image.Image:
        """Create blank page with scanning artifacts (no annotations)."""
        img = Image.new('RGB', size, (255, 255, 255))
        
        # Add some random artifacts
        draw = ImageDraw.Draw(img)
        
        # Random spots/dust
        for _ in range(random.randint(5, 20)):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            r = random.randint(1, 3)
            color = random.randint(200, 240)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(color, color, color))
        
        # Light noise
        img_array = np.array(img).astype(float)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)


def apply_random_edge_case(img: Image.Image, annotations: List[Dict], 
                           case_type: str = None) -> Tuple[Image.Image, List[Dict]]:
    """
    Apply a random edge case transformation.
    
    Args:
        img: Input image
        annotations: List of annotation dicts with 'text' and 'polygon'
        case_type: Specific case type or None for random
        
    Returns:
        Modified image and annotations
    """
    generator = EdgeCaseGenerator()
    
    if case_type is None:
        case_type = random.choice([
            'partial_scan',
            'extreme_rotation',
            'textured_background'
        ])
    
    if case_type == 'partial_scan':
        return generator.create_partial_scan(img, annotations)
    elif case_type == 'extreme_rotation':
        return generator.create_extreme_rotation(img, annotations)
    elif case_type == 'textured_background':
        return generator.create_textured_background(img, annotations)
    else:
        return img, annotations