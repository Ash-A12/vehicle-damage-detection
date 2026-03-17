import cv2 as cv
import numpy as np

# Read image
img = cv.imread("photos/car12.jpg")
cv.imshow("Original", img)

# Create mask
mask = np.zeros(img.shape[:2], np.uint8)

# Models used internally by GrabCut
bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

# Rectangle around the object
rect = (10, 10, img.shape[1]-20, img.shape[0]-20)

# Apply GrabCut
cv.grabCut(img, mask, rect, bgModel, fgModel, 10, cv.GC_INIT_WITH_RECT)

# Convert mask to binary
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

# Extract car
car = img * mask2[:,:,np.newaxis]

# Blur background
blur = cv.GaussianBlur(img,(25,25),0)

# Combine
result = np.where(mask2[:,:,None]==1, img, blur)

cv.imshow("Segmented Car", car)
cv.imshow("Blurred Background", result)

# Convert to grayscale
gray = cv.cvtColor(car, cv.COLOR_BGR2GRAY)

# Scratch enhancement
kernel = np.ones((3,3),np.uint8)
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)

cv.imshow("Scratch Enhancement", tophat)

# Edge detection
blur_gray = cv.GaussianBlur(gray,(9,9),0)
edges = cv.Canny(blur_gray,100,200)

cv.imshow("Edges", edges)

# Detect dents
contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

damage = img.copy()

for cnt in contours:
    area = cv.contourArea(cnt)

    if area > 300:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(damage,(x,y),(x+w,y+h),(0,0,255),2)

cv.imshow("Possible Damage", damage)

cv.waitKey(0)
cv.destroyAllWindows()