import cv2
import numpy as np

# ===================== Read and preprocess image =====================
img = cv2.imread("image_2e-12.png")

# Optional Gaussian blur to reduce noise
img_blur = cv2.GaussianBlur(img, (5,5), 0)

# Convert to grayscale
gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

# ===================== Harris Corner Detection =====================
gray_f32 = np.float32(gray)
dst = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.03)
dst = cv2.dilate(dst, None)  # Dilate to make corners more visible

harris_img = img.copy()
harris_img[dst > 0.001 * dst.max()] = [0, 0, 255]  # Red corners

# ===================== Shi-Tomasi Corner Detection =====================
shi_img = img.copy()
corners = cv2.goodFeaturesToTrack(
    gray, maxCorners=100, qualityLevel=0.001, minDistance=10
)
corners = corners.astype(np.int32)
for c in corners:
    x, y = c.ravel()
    cv2.circle(shi_img, (x, y), 3, (0, 255, 0), -1)  # Green corners

# ===================== FAST Corner Detection =====================
fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
kp = fast.detect(gray, None)
fast_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))  # Blue corners

# ===================== Simple Intensity Thresholding =====================
# Create a binary mask for bright regions
_, thresh_img = cv2.threshold(gray, 70, 80, cv2.THRESH_BINARY)  # Only very bright pixels
thresh_img_color = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

# ===================== Canny Edge Detection =====================
# Detect edges in the image
edges = cv2.Canny(gray, 50, 200)
edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# ===================== Show results =====================
cv2.imshow("Harris Corners (Red)", harris_img)
cv2.imshow("Shi-Tomasi Corners (Green)", shi_img)
cv2.imshow("FAST Corners (Blue)", fast_img)
cv2.imshow("Thresholding Bright Cross", thresh_img_color)
cv2.imshow("Canny Edge Detection", edges_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
