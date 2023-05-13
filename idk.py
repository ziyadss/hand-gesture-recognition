import cv2
import numpy as np

# Load image
img = cv2.imread("data/men/3/3_men (125).JPG")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
max_contour = max(contours, key=cv2.contourArea)

# Find the convex hull of the largest contour
hull = cv2.convexHull(max_contour, returnPoints=False)

# Find the defects in the convex hull
defects = cv2.convexityDefects(max_contour, hull)

# Draw the defects on the image
for i in range(defects.shape[0]):
    s, e, f, d = defects[i][0]
    start = tuple(max_contour[s][0])
    end = tuple(max_contour[e][0])
    far = tuple(max_contour[f][0])
    cv2.line(img, start, end, [0, 255, 0], 2)
    cv2.circle(img, far, 5, [0, 0, 255], -1)

# Show the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
