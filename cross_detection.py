import cv2
import numpy as np

# Read input image
img = cv2.imread("image_n.png")
img1=cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===================== Harris =====================
gray_f32 = np.float32(gray)
dst = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)  # Dilate to make corners more visible

harris_img = img.copy()
harris_img[dst > 0.005 * dst.max()] = [0, 0, 255]  # Mark corners in red

# ===================== Shi-Tomasi =====================
shi_img = img.copy()
corners = cv2.goodFeaturesToTrack(
    gray, maxCorners=100, qualityLevel=0.02, minDistance=10
)
corners = corners.astype(np.int32)

for c in corners:
    x, y = c.ravel()
    cv2.circle(shi_img, (x, y), 3, (0, 255, 0), -1)  # Mark corners with green circles

# ===================== FAST =====================
fast = cv2.FastFeatureDetector_create(threshold=75, nonmaxSuppression=True)
kp = fast.detect(gray, None)

fast_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))  # Mark corners in blue

# ===================== Show results =====================
cv2.imshow("Harris Corners (Red)", harris_img)
cv2.imshow("Shi-Tomasi Corners (Green)", shi_img)
cv2.imshow("FAST Corners (Blue)", fast_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
