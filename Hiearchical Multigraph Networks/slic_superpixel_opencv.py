import cv2
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from skimage.measure import regionprops
#read image
img = cv2.imread('HICO_train2015_00000003.jpg') 
# Convert to LAB
src_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB) # convert to LAB

# SLIC
cv_slic = cv2.ximgproc.createSuperpixelSLIC(src_lab,algorithm = cv2.ximgproc.SLICO, 
region_size = 64)
cv_slic.iterate()
retval = cv_slic.getNumberOfSuperpixels() # = 370
label_out = cv_slic.getLabels()
contours_mask = cv_slic.getLabelContourMask()
contours, hierarchy = cv2.findContours(contours_mask,  
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_2 = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3) 

# Measure properties of labeled image regions
regions = regionprops(label_out)
for region in regions:
    x, y = int(region.centroid[1]), int(region.centroid[0])
    cv2.circle(img_2, (x, y), radius=3, color=(255, 0, 0)) 
    # print(region.centroid)
# Scatter centroid of each superpixel
# plt.scatter([x.centroid[1] for x in regions], [y.centroid[0] for y in regions],c = 'red')
# cv2.imwrite("slic.jpg", img_2)
# plt.savefig('center.jpg')
