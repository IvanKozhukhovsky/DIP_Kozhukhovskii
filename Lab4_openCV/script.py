import cv2
import numpy as np
import matplotlib.pyplot as plt


######################## Basic Morphological Operations #######################
I = cv2.imread('im.jpg', cv2.IMREAD_GRAYSCALE) 
_, binary_im = cv2.threshold(I, 200, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
image_transform = cv2.morphologyEx(binary_im, cv2.MORPH_DILATE, kernel, iterations=3)

# Save results
f, ax = plt.subplots(1, 2, figsize=(10,12))

ax[0].imshow(binary_im, cmap='gray')
ax[0].set_title('Original (binary inverted)')
ax[0].axis('off') 

ax[1].imshow(image_transform, cmap='gray')
ax[1].set_title('Morphological dilating')
ax[1].axis('off') 

plt.tight_layout()
plt.savefig('Basic_Morphological_Operations.png')
plt.show()
###############################################################################

############################ Separation Of Objects ############################
image = cv2.imread('graph1.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
eroded = cv2.erode(binary_image, kernel, iterations=20)
dilated = cv2.dilate(eroded, kernel, iterations=10)
result = cv2.bitwise_and(dilated, image)
borders = cv2.dilate(eroded, kernel) - result 

# Save results
f, ax = plt.subplots(1, 3, figsize=(10,12))

ax[0].imshow(binary_image, cmap='gray')
ax[0].set_title('Original (binary inverted)')
ax[0].axis('off') 

ax[1].imshow(result, cmap='gray')
ax[1].set_title('Tranformed (splited image)')
ax[1].axis('off')

ax[2].imshow(borders, cmap='gray')
ax[2].set_title('Borders')
ax[2].axis('off')

plt.tight_layout()
plt.savefig('Separation_Of_Objects.png')
plt.show()
###############################################################################