
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generator.det.layouts import LayoutFactory, LayoutType

def view_all_layouts_grid():
    """Generates a 4x3 grid of all 12 layout types and saves it."""
    
    print("Generating layout grid (4x3)...")
    
    # Get all layout types from the Enum
    layout_types = list(LayoutFactory.LAYOUTS.keys())
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(24, 30))
    axes = axes.flatten()
    
    # Shared dummy data ensures some consistency, though layouts randomize a lot
    data = {
        "items": [
            {"qty": 2, "unit": 25000, "total": 50000},
            {"qty": 1, "unit": 120000, "total": 120000},
            {"qty": 5, "unit": 10000, "total": 50000},
        ],
        "date": "07/02/2026",
        "subtotal": 220000,
        "vat_rate": 8,
        "vat": 17600,
        "grand_total": 237600,
        "payment_method": "Credit Card",
        "invoice_number": "INV-2026-001"
    }
    
    for i, layout_type in enumerate(layout_types):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        try:
            print(f"Rendering {layout_type.value}...")
            layout = LayoutFactory.create(layout_type)
            img = layout.render(data)
            
            ax.imshow(img)
            ax.set_title(layout_type.value, fontsize=20)
            ax.axis('off')
            
            # Draw polygons
            annotations = layout.get_ocr_annotations()
            for ann in annotations:
                poly = ann.get("polygon")
                if poly:
                    # Create a Polygon patch
                    rect = patches.Polygon(poly, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                        
        except Exception as e:
            print(f"Error rendering {layout_type}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            ax.axis('off')

    # Turn off axes for any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_path = project_root / "tests" / "layouts_grid_view.png"
    plt.savefig(output_path, dpi=100)
    print(f"Saved layout grid to: {output_path}")

if __name__ == "__main__":
    view_all_layouts_grid()
