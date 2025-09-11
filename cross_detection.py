import cv2
import numpy as np
import SEMmodel as sm
import matplotlib.pyplot as plt

# ===================== Read and preprocess image =====================
img, pixel_width_x, pixel_width_y, pixels_x, pixels_y, shift_x, shift_y, rotation = sm.real_image()
print("test1")
img = sm.measured_image(img,pixel_width_x,pixel_width_y,5e-13,4e-7)
print("test2")
# Optional Gaussian blur to reduce noise
#img_blur = cv2.GaussianBlur(img, (7,7), 0)

# Convert to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===================== Denoise image =====================
picture_grid_denoised = sm.denoise_image(img)
print("test3")
# ===================== Simple Intensity Thresholding =====================
# Create a binary mask for bright regions
picture_grid_denoised_uint8 = np.clip(picture_grid_denoised, 0, 255).astype(np.uint8)
_, thresh_img = cv2.threshold(picture_grid_denoised_uint8, 60, 70, cv2.THRESH_BINARY)
thresh_img_color = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
print("test4")
# ===================== Show results =====================

# ===================== Plot grayscale histogram =====================
gray = cv2.cvtColor(picture_grid_denoised, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8,5))
plt.hist(picture_grid_denoised.ravel(), bins=256, range=(0, 255), color='gray')
plt.title('Grayscale Histogram of Denoised SEM Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plotting the denoised
plt.figure(figsize=(12,12))
plt.imshow(picture_grid_denoised)
plt.title('Simulated SEM image denoised test')
plt.colorbar()
plt.tight_layout()
plt.show()
'''''
#cv2.imshow("Thresholding Bright Cross", gray)
cv2.imshow("Thresholding Bright Cross", picture_grid_denoised)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''