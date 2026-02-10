"""
Document Scanner
Integrates 'rembg' (U-2-Net) for robust background removal and document detection.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from rembg import remove

def order_points(pts):
    """
    Orders coordinates: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """
    Applies perspective transform to flatten the document.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped

def enhance_document(image):
    """
    Applies 'Magic Color' effect: CLAHE, denoising, and sharpening.
    """
    # Convert to LAB for adaptive histogram equalization
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Mild denoising
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def find_document_contour_dl(image):
    """
    Deep Learning based contour detection using 'rembg'.
    Returns the 4 corners of the document and the resize ratio.
    """
    height = image.shape[0]
    # Resize image to speed up U-2-Net inference (500px height is optimal)
    ratio = height / 500.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    
    # --- DEEP LEARNING STEP ---
    # Convert BGR (OpenCV) to RGB (rembg expects RGB)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Use rembg to remove background. 
    # This returns an RGBA image where the background is transparent (Alpha=0).
    try:
        no_bg = remove(resized_rgb)
    except Exception as e:
        print(f"Error running rembg: {e}")
        return None, ratio

    # Extract the Alpha channel to use as a binary mask
    # 255 = Foreground (Document), 0 = Background
    mask = no_bg[:, :, 3]
    
    # --- POST-PROCESSING MASK ---
    # Find contours on the clean mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, ratio
    
    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    screen_cnt = None
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        # ApproxPolyDP approximates the curve to a polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # We are looking for a rectangle (4 points)
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    # Fallback: If no perfect 4-point polygon is found, use the bounding box of the largest blob
    if screen_cnt is None and len(contours) > 0:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        screen_cnt = np.int32(box)

    return screen_cnt, ratio, mask # Return mask for visualization

def scan_document(image_path, enhance=True, visualize=False):
    """
    Main pipeline execution.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    orig = image.copy()
    print(f"Processing {image_path} with Deep Learning (rembg)...")
    
    # Detect document
    screen_cnt, ratio, mask = find_document_contour_dl(image)
    
    if screen_cnt is None:
        print("Warning: No document found.")
        result = orig
    else:
        # Scale points back to original image size
        screen_cnt_original = screen_cnt.reshape(4, 2) * ratio
        
        # Warp perspective
        warped = four_point_transform(orig, screen_cnt_original)
        result = warped
        
        if enhance:
            print("Enhancing image...")
            result = enhance_document(result)
            
        
    if visualize and screen_cnt is not None:
        visualize_pipeline(orig, mask, screen_cnt, ratio, result)

def preprocess_image(image_path_or_array, enhance=False):
    """
    API for external scripts to preprocess an image.
    Returns the cropped and enhanced image.
    """
    if isinstance(image_path_or_array, str) or isinstance(image_path_or_array, Path):
        image = cv2.imread(str(image_path_or_array))
        if image is None:
            return None
    else:
        image = image_path_or_array

    orig = image.copy()
    screen_cnt, ratio, mask = find_document_contour_dl(image)
    
    if screen_cnt is None:
        return orig
    
    # Scale points back to original image size
    screen_cnt_original = screen_cnt.reshape(4, 2) * ratio
    
    # Warp perspective
    warped = four_point_transform(orig, screen_cnt_original)
    result = warped
    
    if enhance:
        result = enhance_document(result)
        
    return result

def visualize_pipeline(original, mask, contour, ratio, result):
    """
    Visualizes the mask and final result.
    """
    plt.figure(figsize=(15, 5))
    
    # 1. Detected Contour on Original
    plt.subplot(1, 3, 1)
    vis_orig = original.copy()
    cnt_scaled = (contour.reshape(4, 2) * ratio).astype(np.int32)
    cv2.drawContours(vis_orig, [cnt_scaled], -1, (0, 255, 0), 3)
    plt.title("Detection")
    plt.imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 2. Mask (What rembg sees)
    plt.subplot(1, 3, 2)
    plt.title("Mask (Alpha Channel)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # 3. Final Scanned Result
    plt.subplot(1, 3, 3)
    plt.title("Final Scan")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Document Scanner (rembg + OpenCV)")
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image path')
    parser.add_argument('--visualize', '-v', action='store_true', default=True, help='Visualize steps')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    scan_document(input_path, enhance=False, visualize=args.visualize)

if __name__ == "__main__":
    main()