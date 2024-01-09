import cv2
import os


root = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(root, 'img_0.png'))

_, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
print(kernel)
eroded = cv2.erode(binary_img, kernel, iterations=1)
cv2.imshow('eroded', eroded)
cv2.waitKey(10000)