import cv2
import numpy as np
import SEMmodel as sm
import matplotlib.pyplot as plt

# ===================== Read and preprocess image =====================
img, pixel_width_x, pixel_width_y, pixels_x, pixels_y, shift_x, shift_y, rotation = sm.real_image()
print("test1")
img = sm.measured_image(img, pixel_width_x, pixel_width_y, 5e-13, 4e-7)
print("test2")

# ===================== Denoise image =====================
picture_grid_denoised = sm.denoise_image(img)
print("test3")

# Check data type and range
print(f"Data type: {picture_grid_denoised.dtype}")
print(f"Shape: {picture_grid_denoised.shape}")
print(f"Min value: {np.min(picture_grid_denoised)}")
print(f"Max value: {np.max(picture_grid_denoised)}")

# ===================== Process based on data type =====================
if picture_grid_denoised.dtype == np.float32 or picture_grid_denoised.dtype == np.float64:
    # If it's floating point data, normalize to 0-255 range
    if np.max(picture_grid_denoised) > 1.0: #
        # If value range is large, scale directly to 0-255
        picture_grid_denoised_uint8 = np.clip(picture_grid_denoised, 0, 255).astype(np.uint8)
    else:
        # If value range is between 0-1, scale to 0-255
        picture_grid_denoised_uint8 = (picture_grid_denoised * 255).astype(np.uint8)
else:
    # If it's integer type, use directly
    picture_grid_denoised_uint8 = picture_grid_denoised.astype(np.uint8)

# ===================== Display results =====================

# Plot histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.hist(picture_grid_denoised.ravel(), bins=256, range=(0, 1), color='gray', alpha=0.7)
plt.title('Grayscale Histogram of Denoised SEM Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

# Display denoised image
plt.subplot(1, 3, 2)
if len(picture_grid_denoised_uint8.shape) == 2:  # Grayscale image
    plt.imshow(picture_grid_denoised_uint8, cmap='gray')
else:  # Color image
    plt.imshow(picture_grid_denoised_uint8)
plt.title('Denoised SEM Image')
plt.colorbar()

# Display original image
plt.subplot(1, 3, 3)
plt.imshow(img)
plt.title('SEM Image')
plt.colorbar()

plt.tight_layout()
plt.show()