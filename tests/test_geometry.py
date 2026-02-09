"""
Test geometry transformations with simpler visualization.

Generates a grid of transformed images to verify rotation and shift logic.
"""

import sys
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generator.det.layouts import LayoutFactory
from generator.det.geometry import rotate_point
from generator.det.edge_cases import EdgeCaseGenerator

def view_geometry_transformations():
    """Generates a grid of transformed images and saves it."""
    
    print("Generating geometry test grid...")
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 24))
    axes = axes.flatten()
    
    # Create a base image to transform
    layout = LayoutFactory.create_random()
    data = {"grand_total": 123456}
    base_img = layout.render(data)
    base_annotations = layout.get_ocr_annotations()
    
    # Define scenarios
    scenarios = [
        ("Original", lambda i, a: (i, a)),
        ("Rotate 15°", lambda i, a: rotate_image_with_annotations(i, a, 15)),
        ("Rotate 45°", lambda i, a: rotate_image_with_annotations(i, a, 45)),
        ("Rotate 90°", lambda i, a: rotate_image_with_annotations(i, a, 90)),
        ("Rotate -15°", lambda i, a: rotate_image_with_annotations(i, a, -15)),
        ("Rotate -45°", lambda i, a: rotate_image_with_annotations(i, a, -45)),
        ("Rotate 180°", lambda i, a: rotate_image_with_annotations(i, a, 180)),
        ("Partial Scan", lambda i, a: EdgeCaseGenerator.create_partial_scan(i, a)),
        ("Textured BG", lambda i, a: EdgeCaseGenerator.create_textured_background(i, a)),
        ("Extreme Rot (Rand)", lambda i, a: EdgeCaseGenerator.create_extreme_rotation(i, a)),
        ("Mixed Transform", lambda i, a: rotate_image_with_annotations(*EdgeCaseGenerator.create_partial_scan(i, a), 30)),
        ("Blank w/ Noise", lambda i, a: (EdgeCaseGenerator.create_blank_with_artifacts(i.size), [])),
    ]
    
    for idx, (title, func) in enumerate(scenarios):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        try:
            print(f"Processing: {title}")
            # Always start from fresh copy/layout for some, or reuse base for consistency
            if "Original" in title:
                img, anns = base_img.copy(), base_annotations
            else:
                img, anns = func(base_img.copy(), base_annotations)
            
            ax.imshow(img)
            ax.set_title(title, fontsize=16)
            ax.axis('off')
            
            # Draw polygons
            for ann in anns:
                poly = ann.get("polygon")
                if poly:
                    rect = patches.Polygon(poly, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    
        except Exception as e:
            print(f"Error in {title}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            ax.axis('off')

    plt.tight_layout()
    output_path = project_root / "tests" / "geometry_test_grid.png"
    plt.savefig(output_path, dpi=100)
    print(f"Saved geometry test grid to: {output_path}")


def rotate_image_with_annotations(img, annotations, angle):
    """Helper to rotate image and annotations."""
    # Rotate image with expand (PIL rotates CCW)
    rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    
    # Calculate coordinate transform
    w, h = img.size
    new_w, new_h = rotated.size
    center = (w / 2, h / 2)
    offset_x = (new_w - w) / 2
    offset_y = (new_h - h) / 2
    
    new_anns = []
    for ann in annotations:
        poly = ann['polygon']
        new_poly = []
        for p in poly:
            # Rotate point
            rx, ry = rotate_point((p[0], p[1]), center, angle)
            # Add offset
            new_poly.append([rx + offset_x, ry + offset_y])
            
        new_anns.append({
            "text": ann.get("text", ""),
            "polygon": new_poly
        })
        
    return rotated, new_anns

if __name__ == "__main__":
    view_geometry_transformations()
