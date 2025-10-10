import numpy as np
import matplotlib.pyplot as plt
import cv2
from Denoise_functions import *
from Variables_and_constants import *

def detect_and_plot_harris_corners(
    denoised_grid, 
    block_size=2, 
    ksize=3, 
    k=0.04, 
    threshold_ratio=0.01, 
    dot_radius=1, 
    dot_alpha=0.5,  # Transparency: 0 (invisible) to 1 (solid)
    percentile=50
):
    """
    Perform Harris Corner Detection on a grayscale image and plot it with transparent red dots on corners.

    Parameters:
        denoised_grid (np.ndarray): Grayscale input image as a 2D NumPy array.
        block_size (int): Neighborhood size for corner detection.
        ksize (int): Aperture parameter for the Sobel operator.
        k (float): Harris detector free parameter.
        threshold_ratio (float): Threshold relative to max corner response (between 0 and 1).
        dot_radius (int): Radius of the red dots.
        dot_alpha (float): Opacity of the red dots (0 = transparent, 1 = opaque).
    """
    denoised_grid = denoised_grid/np.max(denoised_grid)
    threshold_value = np.max(denoised_grid)*percentile
    # Step 2: Apply thresholding
    denoised_grid = (denoised_grid >= threshold_value).astype(np.uint8)

    # Step 1: Convert to float32 for Harris
    gray = np.float32(denoised_grid)

    # Step 2: Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    dst = cv2.dilate(dst, None)

    # Step 3: Thresholding
    threshold = threshold_ratio * dst.max()
    corner_mask = dst > threshold

    # Step 4: Scale image properly
    if denoised_grid.max() <= 1.0:
        display_img = (denoised_grid * 255).astype(np.uint8)
    else:
        display_img = denoised_grid.astype(np.uint8)

    # Convert grayscale to color
    base_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # Step 5: Create overlay for drawing transparent dots
    overlay = base_img.copy()

    y_coords, x_coords = np.where(corner_mask)
    for x, y in zip(x_coords, y_coords):
        cv2.circle(overlay, (x, y), radius=dot_radius, color=(0, 0, 255), thickness=-1)

    # Step 6: Blend base image with overlay (transparent red dots)
    blended = cv2.addWeighted(overlay, dot_alpha, base_img, 1 - dot_alpha, 0)

    # Step 7: Display
    if show_plots == True:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
    return denoised_grid

def cross_position(image,threshold):
    
    image = denoise_image(image) # Denoise image
    
    threshold_value = np.max(image)*threshold # Calculate threshold value
    x_coords, y_coords = np.where(image > threshold_value) # Find coordinates where array values exceed the threshold

    if len(x_coords) != 0:
        # Calculate the center of the cross (in pixels)
        cross_center = np.array([np.mean(x_coords),np.mean(y_coords)])
    else:
        print("No points found above threshold.")
    
    return cross_center