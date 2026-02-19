"""
Visual Defects Simulator for realistic document scanning artifacts.

Provides comprehensive defects simulation:
    - Biological/Organic: Stains (coffee, water, oil), Fingerprints
    - Physical Damage: Folds, Creases, Tears, Crumpling
    - Paper Artifacts: Staple holes, Punch holes, Texture
    - Printing/Toner: Fading, Uneven toner, Streaks, Blotches
    - Scanner/Digital: Noise, Blur, Motion blur, Skew, Perspective, Lighting/Shadows
    - Annotations: Handwritten marks, Scribbles, Highlight
"""

import random
import math
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


class DefectSimulator:
    """Apply realistic visual defects to document images."""

    @staticmethod
    def apply_stain(img: Image.Image, intensity: float = 0.3) -> Image.Image:
        """Apply coffee/water stain effect."""
        width, height = img.size
        # Work on copy to avoid modifying original if needed, though we return new image
        img = img.copy()
        img_array = np.array(img).astype(float)
        
        # Random stain location
        x = random.randint(int(width * 0.1), int(width * 0.9))
        y = random.randint(int(height * 0.1), int(height * 0.9))
        radius = random.randint(20, 80)
        
        # Create stain mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw irregular stain shape with multiple circles
        for _ in range(random.randint(3, 7)):
            offset_x = random.randint(-radius//2, radius//2)
            offset_y = random.randint(-radius//2, radius//2)
            r = random.randint(radius//2, radius)
            draw.ellipse([x + offset_x - r, y + offset_y - r, 
                         x + offset_x + r, y + offset_y + r], fill=255)
        
        # Blur for natural look
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
        mask_array = np.array(mask) / 255.0
        
        # Apply brownish/yellowish tint
        stain_color = np.array([160, 140, 100])  # Coffee/Tea color
        if random.random() < 0.3:
            stain_color = np.array([200, 200, 200]) # Grease/Water (darker/translucent)

        for c in range(3):
            # Blend: original * (1 - alpha) + color * alpha
            img_array[:, :, c] = img_array[:, :, c] * (1 - intensity * mask_array) + \
                                 stain_color[c] * intensity * mask_array
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_crease(img: Image.Image, num_creases: int = 1) -> Image.Image:
        """Apply fold/crease effect with visible shadow and highlight."""
        import random
        import math
        
        img_array = np.array(img).astype(float)
        height, width = img_array.shape[:2]
        
        for _ in range(num_creases):
            # Random crease line
            is_horizontal = random.random() < 0.5
            
            if is_horizontal:  # Horizontal crease
                y = random.randint(int(height * 0.2), int(height * 0.8))
                
                for x in range(width):
                    # Wavy variation for natural look
                    offset = int(math.sin(x / 25.0) * 2)
                    center_y = y + offset
                    
                    # Apply gradient around crease - wider and stronger
                    crease_width = 12  # Wider effect
                    
                    for dy in range(-crease_width, crease_width + 1):
                        pixel_y = center_y + dy
                        if 0 <= pixel_y < height:
                            # Normalized distance from crease center (-1 to 1)
                            dist = dy / crease_width
                            
                            # Crease profile: dark in middle (valley), bright on sides (ridges)
                            # Using Gaussian-like curve for dark valley + bright edges
                            valley_darkness = 0.7 * math.exp(-(dist ** 2) / 0.3)  # Dark center
                            ridge_brightness = 0.15 * math.exp(-((abs(dist) - 0.7) ** 2) / 0.1)  # Bright edges
                            
                            factor = 1.0 - valley_darkness + ridge_brightness
                            img_array[pixel_y, x] *= factor
                            
            else:  # Vertical crease
                x = random.randint(int(width * 0.2), int(width * 0.8))
                
                for y in range(height):
                    offset = int(math.sin(y / 25.0) * 2)
                    center_x = x + offset
                    
                    crease_width = 12
                    
                    for dx in range(-crease_width, crease_width + 1):
                        pixel_x = center_x + dx
                        if 0 <= pixel_x < width:
                            dist = dx / crease_width
                            
                            valley_darkness = 0.7 * math.exp(-(dist ** 2) / 0.3)
                            ridge_brightness = 0.15 * math.exp(-((abs(dist) - 0.7) ** 2) / 0.1)
                            
                            factor = 1.0 - valley_darkness + ridge_brightness
                            img_array[y, pixel_x] *= factor
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    @staticmethod
    def apply_crumple(img: Image.Image, strength: float = 0.5) -> Image.Image:
        """Simulate crumpled paper using localized mesh warping or noise."""
        width, height = img.size
        # Simplified crumple using Perlin-like noise for displacement or shading
        # Generating low-freq noise
        noise_size = (width // 10, height // 10)
        noise = np.random.normal(0, 1, (noise_size[1], noise_size[0]))
        noise_img = Image.fromarray((noise * 128 + 128).astype(np.uint8)).resize((width, height), Image.BILINEAR)
        noise_arr = np.array(noise_img).astype(float) / 255.0 # 0.0 to 1.0
        
        # Apply shading based on noise gradient
        img_arr = np.array(img).astype(float)
        # Gradient approximates surface normal
        gy, gx = np.gradient(noise_arr)
        shading = 1.0 + (gx + gy) * strength # Brighten/Darken
        
        for c in range(3):
            img_arr[:,:,c] *= shading
            
        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_shadow(img: Image.Image, intensity: float = 0.4) -> Image.Image:
        """Apply strong scanner shadow or uneven lighting."""
        width, height = img.size
        img_array = np.array(img).astype(float)
        
        # Create gradient mask
        mask = np.ones((height, width))
        mode = random.choice(['linear', 'radial', 'corner'])
        
        if mode == 'linear':
            if random.random() < 0.5:  # Horizontal
                direction = random.choice([1, -1])
                gradient = np.linspace(0, 1, width)
                if direction == -1: gradient = gradient[::-1]
                mask = mask * gradient[np.newaxis, :]
            else:  # Vertical
                direction = random.choice([1, -1])
                gradient = np.linspace(0, 1, height)
                if direction == -1: gradient = gradient[::-1]
                mask = mask * gradient[:, np.newaxis]
                
        elif mode == 'radial':
            # Vignette
             Y, X = np.ogrid[:height, :width]
             center_x, center_y = width/2, height/2
             dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
             max_dist = math.sqrt((width/2)**2 + (height/2)**2)
             mask = 1 - (dist_from_center / max_dist)
             
        elif mode == 'corner':
             # Dark corner
             Y, X = np.ogrid[:height, :width]
             corner_x = random.choice([0, width])
             corner_y = random.choice([0, height])
             dist = np.sqrt((X - corner_x)**2 + (Y - corner_y)**2)
             max_dist = math.sqrt(width**2 + height**2)
             mask = 1 - (dist / (max_dist * 0.7))

        # Normalize mask to 0-1
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        
        # Apply shadow: lessen intensity means stick closer to original, high intensity means darker
        # Final pixel = pixel * (1 - intensity * (1 - mask))
        # Where mask=1 is bright, mask=0 is dark
        shadow_map = 1.0 - intensity * (1.0 - mask)
        
        for c in range(3):
            img_array[:, :, c] *= shadow_map
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_noise(img: Image.Image, intensity: float = 0.05) -> Image.Image:
        """Add Gaussian or ISO noise."""
        img_array = np.array(img).astype(float)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        img_array = img_array + noise
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_salt_pepper_noise(img: Image.Image, amount: float = 0.005) -> Image.Image:
        """Add Salt and Pepper noise (white/black specks)."""
        img_array = np.array(img)
        # Salt
        coords = [np.random.randint(0, i - 1, int(amount * img.size[0] * img.size[1])) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1], :] = 255
        
        # Pepper
        coords = [np.random.randint(0, i - 1, int(amount * img.size[0] * img.size[1])) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1], :] = 0
        
        return Image.fromarray(img_array)

    @staticmethod
    def apply_sand_grain_noise(img: Image.Image, density: float = None,
                               grain_size: int = None) -> Image.Image:
        """
        Simulate sand/grain/speckle noise — clusters of small dark or light dots.
        This is the main false-positive trigger: the model mistakes these clusters for text.
        """
        width, height = img.size
        img_array = np.array(img).astype(np.float32)
        
        if density is None:
            density = random.uniform(0.005, 0.03)
        if grain_size is None:
            grain_size = random.randint(1, 3)
        
        num_grains = int(density * width * height)
        
        for _ in range(num_grains):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            # Random dark or light grain
            if random.random() < 0.7:
                # Dark grain (like sand/dust)
                color_val = random.uniform(0, 80)
            else:
                # Light grain
                color_val = random.uniform(200, 255)
            
            # Draw grain (small square)
            x_end = min(x + grain_size, width)
            y_end = min(y + grain_size, height)
            
            # Partial alpha blend so it doesn't look too artificial
            alpha = random.uniform(0.3, 0.9)
            img_array[y:y_end, x:x_end] = (
                img_array[y:y_end, x:x_end] * (1 - alpha) + color_val * alpha
            )
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
        """Apply blur (Gaussian or Motion)."""
        if random.random() < 0.7:
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            # Simple motion blur simulation using numpy/scipy
            kernel_size = int(radius * 2) + 1
            # Ensure minimum valid kernel size
            kernel_size = max(kernel_size, 3)
            # Ensure odd number
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            center = kernel_size // 2

            # Horizontal or Vertical motion
            if random.random() < 0.5:
                kernel[center, :] = 1.0 / kernel_size
            else:
                kernel[:, center] = 1.0 / kernel_size

            # Apply convolution using scipy if available, otherwise use PIL
            try:
                from scipy import ndimage
                img_array = np.array(img).astype(np.float32)
                blurred = np.zeros_like(img_array)
                for c in range(img_array.shape[2]):
                    blurred[:, :, c] = ndimage.convolve(img_array[:, :, c], kernel, mode='nearest')
                return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))
            except ImportError:
                # Fallback to Gaussian blur if scipy not available
                return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def apply_local_blur(img: Image.Image, num_zones: int = None) -> Image.Image:
        """
        Apply blur to random horizontal strips (simulates scanner out-of-focus zones).
        This is the key defect for training the model NOT to merge blurry text lines.
        """
        width, height = img.size
        img_array = np.array(img).astype(np.float32)
        
        if num_zones is None:
            num_zones = random.randint(1, 4)
        
        for _ in range(num_zones):
            # Random horizontal strip
            strip_height = random.randint(20, max(30, height // 6))
            y_start = random.randint(0, height - strip_height)
            y_end = y_start + strip_height
            
            # Random blur radius (can be quite strong)
            radius = random.uniform(1.0, 4.0)
            
            # Extract strip, blur, and paste back
            strip = img.crop((0, y_start, width, y_end))
            strip = strip.filter(ImageFilter.GaussianBlur(radius=radius))
            strip_array = np.array(strip).astype(np.float32)
            
            # Smooth transition: create gradient mask at edges
            fade = min(10, strip_height // 4)
            mask = np.ones(strip_height, dtype=np.float32)
            for i in range(fade):
                t = i / fade
                mask[i] = t
                mask[strip_height - 1 - i] = t
            mask = mask[:, np.newaxis, np.newaxis]
            
            # Blend
            img_array[y_start:y_end] = img_array[y_start:y_end] * (1 - mask) + strip_array * mask
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_broken_text(img: Image.Image, num_breaks: int = None) -> Image.Image:
        """
        Simulate broken/interrupted printing where thin streaks cut through text.
        This creates the 'gãy đoạn' (broken text) effect the user described.
        """
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        if num_breaks is None:
            num_breaks = random.randint(2, 8)
        
        bg_color = (255, 255, 255)  # Assume white background
        
        for _ in range(num_breaks):
            # Thin horizontal white lines (1-3 px) cutting through text
            y = random.randint(0, height - 1)
            thickness = random.randint(1, 3)
            
            # Partial width (not full width to look realistic)
            x_start = random.randint(0, width // 3)
            x_end = random.randint(width * 2 // 3, width)
            
            # Semi-transparent white streak
            for dy in range(thickness):
                if y + dy < height:
                    for x in range(x_start, x_end):
                        if random.random() < 0.7:  # Not every pixel
                            draw.point((x, y + dy), fill=bg_color)
        
        return img

    
    @staticmethod
    def apply_brightness_variation(img: Image.Image, factor: float = None) -> Image.Image:
        """Vary brightness globally."""
        if factor is None:
            factor = random.choice([random.uniform(0.6, 0.9), random.uniform(1.1, 1.4)])
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_contrast_variation(img: Image.Image, factor: float = None) -> Image.Image:
        """Vary contrast globally."""
        if factor is None:
            factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
        
    @staticmethod
    def apply_toner_loss(img: Image.Image, strength: float = 0.5) -> Image.Image:
        """Simulate running out of ink/toner (fading strips)."""
        width, height = img.size
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # Vertical strips of fading
        num_strips = random.randint(3, 8)
        for _ in range(num_strips):
            x = random.randint(0, width)
            w = random.randint(10, 100)
            pass
            
        # Simpler implementation: Perlin-ish noise for uneven toner
        img_array = np.array(img).astype(float)
        
        # Generate low freq noise
        noise = np.random.rand(height // 20, width // 20)
        noise = np.array(Image.fromarray((noise*255).astype(np.uint8)).resize((width, height), Image.BILINEAR)) / 255.0
        
        # Threshold: if noise < threshold, lighten pixel (faded)
        # Fading means moving towards 255
        fade_map = (noise < strength).astype(float) * random.uniform(0.3, 0.7) # factor
        
        for c in range(3):
            # img + (255 - img) * fade_factor
            img_array[:, :, c] = img_array[:, :, c] + (255 - img_array[:, :, c]) * fade_map
            
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_handwritten_mark(img: Image.Image) -> Image.Image:
        """Add random scribble or checkmark."""
        width, height = img.size
        overlay = Image.new('RGBA', (width, height), (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        color = random.choice(['blue', 'black', 'red'])
        pen_width = random.randint(1, 3)
        
        # Random location
        x = random.randint(int(width*0.1), int(width*0.9))
        y = random.randint(int(height*0.1), int(height*0.9))
        
        mark_type = random.choice(['check', 'circle', 'scribble', 'cross'])
        
        if mark_type == 'check':
            points = [(x, y), (x+10, y+20), (x+30, y-10)]
            draw.line(points, fill=color, width=pen_width)
        elif mark_type == 'cross':
             draw.line([(x, y), (x+20, y+20)], fill=color, width=pen_width)
             draw.line([(x+20, y), (x, y+20)], fill=color, width=pen_width)
        elif mark_type == 'circle':
             draw.ellipse([x, y, x+40, y+30], outline=color, width=pen_width)
        elif mark_type == 'scribble':
             points = [(x + i*5 + random.randint(-2,2), y + random.randint(-5,5)) for i in range(10)]
             draw.line(points, fill=color, width=pen_width)
             
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    @staticmethod
    def apply_staple_holes(img: Image.Image) -> Image.Image:
        """Add staple holes or punch holes."""
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        hole_type = random.choice(['staple', 'punch'])
        
        if hole_type == 'punch':
             # 2-hole punch on left
             r = 15
             x = 20
             y1 = height // 3
             y2 = 2 * height // 3
             for y in [y1, y2]:
                 # Draw hole (dark circle inside, light highlight edge)
                 draw.ellipse([x-r, y-r, x+r, y+r], fill=(30, 30, 30))
                 draw.arc([x-r, y-r, x+r, y+r], start=0, end=360, fill=(200, 200, 200), width=1)
        else:
             # Staple top left
             x, y = 30, 30
             w, h = 10, 3 # Horizontal staple or vertical
             angle = random.randint(-45, 45)
             
             # Visualize staple as thin silver rect + small holes
             # Simplify: just two small holes
             dist = 15
             r = 2
             # Rotate points
             # Just draw two dots
             draw.ellipse([x, y, x+r*2, y+r*2], fill=(20, 20, 20))
             draw.ellipse([x+dist, y+dist/2, x+dist+r*2, y+dist/2+r*2], fill=(20, 20, 20))
             # Draw staple wire
             draw.line([x+r, y+r, x+dist+r, y+dist/2+r], fill=(180, 180, 180), width=2)
             
        return img
    
    @staticmethod
    def apply_tear(img: Image.Image, 
                   edge: str = None,
                   tear_width: int = None,
                   tear_depth: int = None) -> Image.Image:
        """
        Simulate torn edge effect by removing a jagged section from edges or corners.

        Args:
            img: Input PIL Image
            edge: Which edge to tear ('top', 'bottom', 'left', 'right', 'corner_tl', 
                  'corner_tr', 'corner_bl', 'corner_br'). Random if None.
            tear_width: Width of the tear area. Random if None.
            tear_depth: How deep the tear goes. Random if None.

        Returns:
            PIL Image with torn edge effect
        """
        import random
        import math

        width, height = img.size
        img_array = np.array(img).copy()

        # Random selection if not specified
        if edge is None:
            edge = random.choice(['top', 'bottom', 'left', 'right', 
                                 'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'])

        # Create mask for tear region
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Generate jagged edge points
        def generate_jagged_line(start, end, num_points=15, roughness=8):
            """Generate a jagged line between two points."""
            points = []
            for i in range(num_points + 1):
                t = i / num_points
                # Linear interpolation
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                # Add perpendicular noise for jagged effect
                noise = random.uniform(-roughness, roughness)
                # Perpendicular direction
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    perp_x = -dy / length * noise
                    perp_y = dx / length * noise
                    x += perp_x
                    y += perp_y
                points.append((int(x), int(y)))
            return points

        # Set default dimensions
        if tear_width is None:
            tear_width = random.randint(80, min(200, width // 2))
        if tear_depth is None:
            tear_depth = random.randint(40, min(120, height // 3))

        polygon_points = []

        if edge == 'top':
            # Tear from top edge - creates a bite from the top
            start_x = random.randint(0, width - tear_width)
            # Define corners of the tear region
            top_left = (start_x, 0)
            top_right = (start_x + tear_width, 0)
            bottom_right = (start_x + tear_width, tear_depth)
            bottom_left = (start_x, tear_depth)

            # Create jagged bottom edge (the tear line)
            jagged_edge = generate_jagged_line(bottom_right, bottom_left, num_points=12)

            # Build polygon: top edge → right edge → jagged bottom → left edge
            polygon_points = [top_left, top_right] + jagged_edge + [top_left]

        elif edge == 'bottom':
            # Tear from bottom edge
            start_x = random.randint(0, width - tear_width)
            bottom_left = (start_x, height)
            bottom_right = (start_x + tear_width, height)
            top_right = (start_x + tear_width, height - tear_depth)
            top_left = (start_x, height - tear_depth)

            jagged_edge = generate_jagged_line(top_left, top_right, num_points=12)
            polygon_points = [bottom_left, bottom_right] + jagged_edge + [bottom_left]

        elif edge == 'left':
            # Tear from left edge
            start_y = random.randint(0, height - tear_width)
            top_left = (0, start_y)
            bottom_left = (0, start_y + tear_width)
            bottom_right = (tear_depth, start_y + tear_width)
            top_right = (tear_depth, start_y)

            jagged_edge = generate_jagged_line(bottom_right, top_right, num_points=12)
            polygon_points = [top_left, bottom_left] + jagged_edge + [top_left]

        elif edge == 'right':
            # Tear from right edge
            start_y = random.randint(0, height - tear_width)
            top_right = (width, start_y)
            bottom_right = (width, start_y + tear_width)
            bottom_left = (width - tear_depth, start_y + tear_width)
            top_left = (width - tear_depth, start_y)

            jagged_edge = generate_jagged_line(bottom_left, top_left, num_points=12)
            polygon_points = [top_right, bottom_right] + jagged_edge + [top_right]

        elif edge == 'corner_tl':
            # Top-left corner tear (diagonal)
            corner_size = random.randint(60, min(150, width // 3, height // 3))
            corner = (0, 0)
            right = (corner_size + random.randint(-10, 10), 0)
            bottom = (0, corner_size + random.randint(-10, 10))

            # Jagged arc from right to bottom
            jagged_edge = generate_jagged_line(right, bottom, num_points=10, roughness=10)
            polygon_points = [corner, right] + jagged_edge + [corner]

        elif edge == 'corner_tr':
            # Top-right corner tear
            corner_size = random.randint(60, min(150, width // 3, height // 3))
            corner = (width, 0)
            left = (width - corner_size + random.randint(-10, 10), 0)
            bottom = (width, corner_size + random.randint(-10, 10))

            jagged_edge = generate_jagged_line(bottom, left, num_points=10, roughness=10)
            polygon_points = [corner, left] + jagged_edge + [corner]

        elif edge == 'corner_bl':
            # Bottom-left corner tear
            corner_size = random.randint(60, min(150, width // 3, height // 3))
            corner = (0, height)
            right = (corner_size + random.randint(-10, 10), height)
            top = (0, height - corner_size + random.randint(-10, 10))

            jagged_edge = generate_jagged_line(right, top, num_points=10, roughness=10)
            polygon_points = [corner, right] + jagged_edge + [corner]

        elif edge == 'corner_br':
            # Bottom-right corner tear
            corner_size = random.randint(60, min(150, width // 3, height // 3))
            corner = (width, height)
            left = (width - corner_size + random.randint(-10, 10), height)
            top = (width, height - corner_size + random.randint(-10, 10))

            jagged_edge = generate_jagged_line(top, left, num_points=10, roughness=10)
            polygon_points = [corner, top] + jagged_edge + [corner]

        # Draw the tear polygon
        if polygon_points:
            draw.polygon(polygon_points, fill=255)

            # Optional: Add slight blur to tear edge for realism
            mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5))

            # Apply mask - white out the torn region
            mask_array = np.array(mask) > 128
            img_array[mask_array] = [255, 255, 255]  # White background

            # Optional: Add subtle shadow along tear edge for depth
            edge_mask = np.array(mask)
            # Dilate slightly to create shadow region
            from scipy import ndimage
            if 'scipy' in sys.modules:
                shadow_mask = ndimage.binary_dilation(edge_mask > 128, iterations=2)
                shadow_mask = shadow_mask & ~(edge_mask > 128)
                # Darken slightly
                img_array[shadow_mask] = (img_array[shadow_mask] * 0.9).astype(np.uint8)

        return Image.fromarray(img_array)

    @classmethod
    def apply_random_defects(cls, img: Image.Image, 
                            probability: float = 0.5,
                            num_defects: int = None) -> Image.Image:
        """Apply random combination of defects."""
        defect_functions = [
            (cls.apply_stain, {'intensity': random.uniform(0.1, 0.4)}),
            (cls.apply_crease, {'num_creases': random.randint(1, 3)}),
            (cls.apply_shadow, {'intensity': random.uniform(0.1, 0.5)}),
            (cls.apply_noise, {'intensity': random.uniform(0.02, 0.08)}),
            (cls.apply_salt_pepper_noise, {'amount': random.uniform(0.002, 0.01)}),
            (cls.apply_sand_grain_noise, {}),
            (cls.apply_blur, {'radius': random.uniform(0.5, 3.0)}),
            (cls.apply_local_blur, {}),
            (cls.apply_broken_text, {}),
            (cls.apply_brightness_variation, {}),
            (cls.apply_contrast_variation, {}),
            (cls.apply_crumple, {'strength': random.uniform(0.3, 0.7)}),
            (cls.apply_toner_loss, {'strength': random.uniform(0.3, 0.6)}),
            (cls.apply_handwritten_mark, {}),
            (cls.apply_staple_holes, {}),
            # Tear is distinct, keeping it less frequent 
        ]
        
        # Create a weighted choice or ensure basics are picked
        if num_defects is not None:
            # Apply exactly num_defects random defects
            selected = random.sample(defect_functions, min(num_defects, len(defect_functions)))
            for func, kwargs in selected:
                img = func(img, **kwargs)
        else:
            # Apply each defect with given probability
            for func, kwargs in defect_functions:
                if random.random() < probability:
                    img = func(img, **kwargs)
        
        # Occasional tear
        if random.random() < 0.05:
            img = cls.apply_tear(img)
            
        return img


# Helper functions to maintain compatibility
def apply_defects_light(img: Image.Image) -> Image.Image:
    """Apply light defects (1-2 minimal defects)."""
    # Restrict to subtle defects
    functions = [
        (DefectSimulator.apply_noise, {'intensity': 0.01}),
        (DefectSimulator.apply_blur, {'radius': 0.5}),
        (DefectSimulator.apply_brightness_variation, {}),
        (DefectSimulator.apply_shadow, {'intensity': 0.1}),
        (DefectSimulator.apply_crease, {'num_creases': 1}),
    ]
    func, kwargs = random.choice(functions)
    img = func(img, **kwargs)
    if random.random() < 0.3:
        func, kwargs = random.choice(functions)
        img = func(img, **kwargs)
    return img


def apply_defects_medium(img: Image.Image) -> Image.Image:
    """Apply medium defects (2-4 defects)."""
    return DefectSimulator.apply_random_defects(img, num_defects=random.randint(2, 4))


def apply_defects_heavy(img: Image.Image) -> Image.Image:
    """Apply heavy defects (lots of damage)."""
    img = DefectSimulator.apply_random_defects(img, num_defects=random.randint(4, 7))
    # Almost guarantee a stain or heavy shadow
    if random.random() < 0.5:
        img = DefectSimulator.apply_stain(img, intensity=0.5)
    return img