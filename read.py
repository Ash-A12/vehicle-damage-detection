import cv2 as cv
import numpy as np

# 1. Load and Isolate
img = cv.imread("photos/car14.jpeg")
if img is None:
    print("Error: Image not found.")
    exit()

# Create mask for GrabCut
mask = np.zeros(img.shape[:2], np.uint8)
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)

# Keep the rectangle slightly away from the extreme edges to avoid border noise
rect = (15, 15, img.shape[1]-30, img.shape[0]-30)
cv.grabCut(img, mask, rect, bgModel, fgModel, 5, cv.GC_INIT_WITH_RECT)

# Create a binary mask where 1 and 3 are foreground
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
car_only = img * mask2[:, :, np.newaxis]

# 2. Pre-processing for Dents
gray = cv.cvtColor(car_only, cv.COLOR_BGR2GRAY)

# Apply CLAHE to equalize light (dents are often hidden in shadows/glare)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

# 3. Morphological Filtering
# We use Black-Hat to find dark deformations (dents) against the paint
# Use a larger kernel to capture the scale of a dent vs a tiny scratch
kernel_dent = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
blackhat = cv.morphologyEx(enhanced_gray, cv.MORPH_BLACKHAT, kernel_dent)

# Threshold the result to get distinct blobs
_, thresh = cv.threshold(blackhat, 15, 255, cv.THRESH_BINARY)

# 4. Clean up Noise
# Remove very small specs and join nearby damaged areas
kernel_clean = np.ones((5, 5), np.uint8)
cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_clean)
dilated = cv.dilate(cleaned, kernel_clean, iterations=2)

# 5. Intelligent Contour Filtering
contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
output = img.copy()

for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 150:  # Filter out tiny noise
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # LOGIC: Dents are usually somewhat "boxy" or "round". 
        # Door seams/scratches are very thin (very high or low aspect ratio).
        if 0.2 < aspect_ratio < 5.0:
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(output, "Potential Dent", (x, y - 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display Results
cv.imshow("Enhanced Gray (CLAHE)", enhanced_gray)
cv.imshow("BlackHat Transform", blackhat)
cv.imshow("Detected Damage", output)

cv.waitKey(0)
cv.destroyAllWindows()