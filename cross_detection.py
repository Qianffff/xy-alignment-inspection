import cv2
import numpy as np
import SEMmodel as sm
import matplotlib.pyplot as plt

# ===================== Read and preprocess image =====================
img = cv2.imread("image_2e-12.png")
img, pixel_width_x, pixel_width_y, pixels_x, pixels_y, shift_x, shift_y, rotation = sm.real_image()

img = sm.measured_image(img,5e-12,4e-7)

# Optional Gaussian blur to reduce noise
#img_blur = cv2.GaussianBlur(img, (7,7), 0)

# Convert to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===================== Denoise image =====================
picture_grid_denoised = sm.denoise_image(img)

# ===================== Simple Intensity Thresholding =====================
# Create a binary mask for bright regions
picture_grid_denoised_uint8 = np.clip(picture_grid_denoised, 0, 255).astype(np.uint8)
_, thresh_img = cv2.threshold(picture_grid_denoised_uint8, 60, 70, cv2.THRESH_BINARY)
thresh_img_color = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

# ===================== Show results =====================
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