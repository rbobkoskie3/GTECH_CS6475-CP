# ASSIGNMENT 4
# Your Name
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

def gradientX(image):
    """ This function differentiates an image in the X direction.

    Note: See lectures 02-06 (Differentiating an image in X and Y) for a good
    explanation of how to perform this operation.

    No calls to library functions or use of vectorized operations is permitted.

    The X direction means that you are subtracting columns:
    der. F(x, y) = F(x+1, y) - F(x, y)
    This corresponds to image[r,c] = image[r,c+1] - image[r,c]

    For the last column, c+1 will be out of bounds.  In this case, you may
    simply set the value of the gradient to zero.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the X direction stored as 
        a CV_16S (np.int16) .
    """
    # WRITE YOUR CODE HERE.
    gradX_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int16)
    image = image.astype(np.int16)

    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1] - 1):
            pix = (image.item(y, x + 1) - image.item(y, x))
            gradX_image.itemset(y, x, pix)

    gradX_image = gradX_image.astype(np.int16)
    #cv2.imshow('GRADX IMAGE', gradX_image)
    #cv2.waitKey()
    #cv2.imwrite('GRADX_IMAGE.jpg', gradX_image)

    return gradX_image
    # END OF FUNCTION.

def gradientY(image):
    """ This function differentiates an image in the Y direction.

    Note: See lectures 02-06 (Differentiating an image in X and Y) for a good
    explanation of how to perform this operation.

    No calls to library functions or use of vectorized operations is permitted.

    The Y direction means that you are subtracting rows:
    der. F(x, y) = F(x, y+1) - F(x, y)
    This corresponds to image[r,c] = image[r+1,c] - image[r,c]

    For the last row, r+1 will be out of bounds.  In this case, you may
    simply set the value of the gradient to zero.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the Y direction stored as 
        a CV_16S (np.int16) .
    """
    # WRITE YOUR CODE HERE.
    gradY_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int16)
    image = image.astype(np.int16)
    #print gradY_image.dtype
    gradY_image = gradY_image.astype(np.int16)
    #print gradY_image.dtype

    for x in range(0, image.shape[1]):
        for y in range(0, image.shape[0] - 1):
            pix = (image.item(y + 1, x) - image.item(y, x))
            gradY_image.itemset(y, x, pix)

    gradY_image = gradY_image.astype(np.int16)
    #print gradY_image.dtype
    #cv2.imshow('GRADY IMAGE', gradY_image)
    #cv2.waitKey()
    #cv2.imwrite('GRADY_IMAGE.JPG', gradY_image)

    return gradY_image
    # END OF FUNCTION.

def easyFilter(image, kernel):
    """ This function applies the kernel to the input image

    The input image may be assumed to have type CV_8U (np.uint8)
    and the output image should have type CV_16S (np.uint16).

    No calls to library function or use of vectorized operations is permitted.

    The function should be equivalent to calling 

    cv2.filter2D(image, CV_16S, kernel, achor=(-1,-1), delta=0, borderType=cv2.BORDER_REFLECT_101)

    See the [OpenCV documentation](http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d)
    for details.
    
    You may assume that the kernel has an odd size in both dimensions.

    """
    
    # WRITE YOUR CODE HERE.
    '''
    ###########################################
    # BEGIN UNIT TEST CONFIG
    ###########################################
    kernel = np.array([[-1, 0, 1, 1, 1],
                       [-2, 0, 2, 1, 1],
                       [-1, 0, 1, 1, 1],
                       [-1, 0, 1, 1, 1],
                       [-1, 0, 1, 1, 1]], dtype=np.int16)

    image = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])

    image = np.array([[1, 0, 3, 4, 5],
                     [0, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15],
                     [16, 17, 18, 19, 20],
                     [21, 22, 23, 24, 25]])

    image = np.array([[1, 0, 3, 4, 5, 1],
                     [0, 7, 8, 9, 10, 2],
                     [11, 12, 13, 14, 15, 3],
                     [16, 17, 18, 19, 20, 4],
                     [21, 22, 23, 24, 25, 5],
                     [21, 22, 23, 24, 25, 6]])

    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print '\n', image

    testNumpyMB = cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    #print '\n', testNumpyMB

    image = image.astype(np.int16)
    ref_img = cv2.filter2D(image, cv2.CV_16S, kernel, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_REFLECT_101)
    print '\n', ref_img

    ###########################################
    # END UNIT TEST CONFIG
    ###########################################
    '''

    #EF_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int16)
    #np.copyto(EF_image, image)
    pad = (kernel.shape[0] - 1) / 2
    image = image.astype(np.int16)

    # Insert  row(s)
    count = 1
    for x in range(pad):
        pix_row = []
        for pix in image[x + count]:
            pix_row.append(pix)
        count +=1
        pix_row = [pix_row]
        image = np.insert(image, [0], pix_row, axis=0)

    # Append row(s)
    count = 2
    for x in range(pad):
        pix_row = []
        for pix in image[image.shape[0] - x - count]:
            pix_row.append(pix)
        count +=1
        pix_row = [pix_row]
        image = np.append(image, pix_row, axis=0)

    # Insert  columns(s)
    count = 1
    for x in range(pad):
        pix_col = []
        for pix in image[:, x + count]:
            pix_col.append(pix)
        count +=1
        pix_col = [pix_col]
        pix_col = np.array(pix_col)
        image = np.insert(image, [0], pix_col.T, axis=1)

    # Append columns(s)
    count = 2
    for x in range(pad):
        pix_col = []
        for pix in image[:, image.shape[1] - x - count]:
            pix_col.append(pix)
        count +=1
        pix_col = [pix_col]
        pix_col = np.array(pix_col)
        image = np.append(image, pix_col.T, axis=1)

    #print '\n', image, type(image)
    #print '\n', testNumpyMB == image

    raterize_img = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int16)
    for x in range(pad, image.shape[0] - pad):
        for y in range(pad, image.shape[1] - pad):
            pix_arr = image[x-pad:x+pad+1, y-pad:y+pad+1]
            pix = np.sum(np.sum(np.multiply(pix_arr, kernel), axis=0))
            raterize_img.itemset(x, y, pix)


    raterize_img = raterize_img[pad:image.shape[0]-pad, pad:image.shape[1]-pad]
    raterize_img = raterize_img.astype(np.int16)
    #print '\n', raterize_img

    return raterize_img
    # END OF FUNCTION.

def main():
    #This code demonstrates some functions that you might use to debug.
    img = cv2.imread(os.path.join("test_images", "butterfly.jpg"), cv2.IMREAD_GRAYSCALE)

    K = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.int16)

    f = easyFilter(img, K)
    ref_f = cv2.filter2D(img, cv2.CV_16S, K, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_REFLECT_101)

    print np.any(cv2.absdiff(f, ref_f))
  
    cv2.normalize(f, f, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(ref_f, ref_f, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow('f', f.astype(np.uint8))
    cv2.imshow('ref_f', ref_f.astype(np.uint8))

    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
