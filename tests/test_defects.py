
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generator.det.defects import DefectSimulator

def create_base_image():
    """Create a simple base image with text."""
    img = Image.new('RGB', (400, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some lines of text
    for i in range(20):
        y = 30 + i * 25
        draw.text((30, y), f"Sample Invoice Line {i+1} - $ {i*10}.00", fill='black')
        
    # Draw a box
    draw.rectangle([50, 400, 350, 550], outline='black', width=2)
    draw.text((60, 410), "Total Amount: $ 1234.56", fill='black')
    
    return img

def test_defects_grid():
    """Generate a grid of all available defects."""
    base_img = create_base_image()
    
    defects = [
        ("Original", lambda x: x),
        ("Stain", lambda x: DefectSimulator.apply_stain(x, intensity=0.5)),
        ("Crease", lambda x: DefectSimulator.apply_crease(x, num_creases=2)),
        ("Crumple", lambda x: DefectSimulator.apply_crumple(x, strength=0.6)),
        ("Shadow", lambda x: DefectSimulator.apply_shadow(x, intensity=0.5)),
        ("Noise", lambda x: DefectSimulator.apply_noise(x, intensity=0.05)),
        ("Salt & Pepper", lambda x: DefectSimulator.apply_salt_pepper_noise(x, amount=0.01)),
        ("Blur", lambda x: DefectSimulator.apply_blur(x, radius=1.5)),
        ("Toner Loss", lambda x: DefectSimulator.apply_toner_loss(x, strength=0.5)),
        ("Handwritten", lambda x: DefectSimulator.apply_handwritten_mark(x)),
        ("Staple Holes", lambda x: DefectSimulator.apply_staple_holes(x)),
        ("Tear", lambda x: DefectSimulator.apply_tear(x)),
        ("Random Context", lambda x: DefectSimulator.apply_random_defects(x, num_defects=3))
    ]
    
    # Plotting
    cols = 4
    rows = (len(defects) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()
    
    for i, (name, func) in enumerate(defects):
        ax = axes[i]
        try:
            # Apply defect
            img_defect = func(base_img.copy())
            ax.imshow(img_defect)
            ax.set_title(name)
            ax.axis('off')
        except Exception as e:
            print(f"Error applying {name}: {e}")
            ax.text(0.5, 0.5, f"Error: {name}", ha='center')
            ax.axis('off')
            
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig('tests/defects_grid.png')
    print("Saved defects grid visualization to tests/defects_grid.png")

if __name__ == "__main__":
    test_defects_grid()
