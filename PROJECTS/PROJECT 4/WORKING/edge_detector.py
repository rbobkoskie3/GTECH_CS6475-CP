# ASSIGNMENT 4
# Robert Bobkoskie
# rbobkoskie3

import os
import cv2
import numpy as np
import scipy as sp

""" Assignment 4 - Detecting Gradients / Edges

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. (This is a problem
    for us when grading because running 200 files results a lot of images being
    saved to file and opened in dialogs, which is not ideal). Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into classes,
    or your own infrastructure. This makes grading very difficult for us. Please
    only write code in the allotted region.
"""

def EdgeDetect(image):

    K = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.int16)

    img = image
    img = cv2.GaussianBlur(img,(5,5),0)
    #img = cv2.medianBlur(img,5)

    # http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
    #laplacian = cv2.Laplacian(img,cv2.CV_64F)
    #sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    #sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

    # http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    #sigma = 0.33
    #v = np.median(img)
    #lower = int(max(0, (1.0 - sigma) * v))
    #upper = int(min(255, (1.0 + sigma) * v))
    #img = cv2.Canny(img, lower, upper)

    img = cv2.filter2D(img, cv2.CV_16S, K, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_REFLECT_101)
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img[img>128] = 255
    img[img<=128] = 0

    cv2.imwrite('EDGE_IMAGE.jpg', img)

def main():

    img = cv2.imread("italy 09 047.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread("italy 09 035.jpg", cv2.IMREAD_GRAYSCALE)

    EdgeDetect(img)

    '''
    K = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.int16)

    ref_f = cv2.filter2D(img, cv2.CV_16S, K, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_REFLECT_101)  
    cv2.normalize(ref_f, ref_f, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('ref_f', ref_f.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    main()

