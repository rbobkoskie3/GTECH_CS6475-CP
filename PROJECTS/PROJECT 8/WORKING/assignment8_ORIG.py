# ASSIGNMENT 8
# Your Name

import numpy as np
import scipy as sp
import scipy.signal
import cv2

# These lines are not required by the vagrant environment, but may be
# required if you are working from a local installation
# Import ORB as SIFT to avoid confusion.
# try:
#     from cv2 import ORB as SIFT
# except ImportError:
#     try:
#         from cv2 import SIFT
#     except ImportError:
#         try:
#             SIFT = cv2.ORB_create
#         except:
#             raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
#                                  % cv2.__version__)

""" Assignment 8 - Panoramas

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

    3. DO NOT change the format of this file. Do not put functions into
    classes, or your own infrastructure. This makes grading very difficult for
    us. Please only write code in the allotted region.

    4. This file has only been tested in the provided Vagrant environment. You
    are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""


def getImageCorners(image):
    """ For an input image, return the x, y coordinates of its four corners. 

    Note: The reasoning for the shape of the array can be explained if you look
    at the documentation for cv2.perspectiveTransform which will be used on the
    output of this function. Since we will apply the homography to the corners
    of the image, it needs to be in that format.

    Another note: When storing your corners, they are assumed to be in the form
    (X, Y) -- keep this in mind and make SURE you get it right.

    Args:
        image (numpy.ndarray): Input can be a grayscale or color image.

    Returns:
        corners (numpy.ndarray): Array of shape (4, 1, 2). Type of values in
                                 the array is np.float32.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE
    '''
    image = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]])
    #corners[0, 0] = image[0, 0]
    #corners = image[::image.shape[0]-1, ::image.shape[1]-1]
    print image
    '''
    corners[::] = [[[0, 0]],
                  [[image.shape[1], 0]],
                  [[0, image.shape[0]]],
                  [[image.shape[1], image.shape[0]]]]

    return corners
    # END OF FUNCTION


def findMatchesBetweenImages(image_1, image_2, num_matches):
    """ Return the top list of matches between two input images.

    Note: You will not be graded for this function. This function is almost
    identical to the function in Assignment 7 (we just parametrized the number
    of matches). We expect you to use the function you wrote in A7 here.

    This function detects and computes SIFT (or ORB) from the input images, and
    returns the best matches using the normalized Hamming Distance through
    brute force matching.

    Args:
        image_1 (numpy.ndarray): The first image (grayscale).
        image_2 (numpy.ndarray): The second image. (grayscale).
        num_matches (int): The number of desired matches. If there are not
                           enough, return as many matches as you can.

    Returns:
        image_1_kp (list): The image_1 keypoints, the elements are of type
                           cv2.KeyPoint.
        image_2_kp (list): The image_2 keypoints, the elements are of type
                           cv2.KeyPoint.
        matches (list): A list of matches, length 'num_matches'. Each item in
                        the list is of type cv2.DMatch. If there are less
                        matches than num_matches, this function will return as
                        many as it can.

    """
    # matches - type: list of cv2.DMath
    matches = None
    # image_1_kp - type: list of cv2.KeyPoint items.
    image_1_kp = None
    # image_1_desc - type: numpy.ndarray of numpy.uint8 values.
    image_1_desc = None
    # image_2_kp - type: list of cv2.KeyPoint items.
    image_2_kp = None
    # image_2_desc - type: numpy.ndarray of numpy.uint8 values.
    image_2_desc = None

    # COPY YOUR CODE FROM A7 HERE. REMEMBER TO MODIFY IT SO THAT IT
    # RETURNS num_matches MATCHES, NOT A FIXED NUMBER OF MATCHES.
    SIFT = cv2.ORB_create
    orb = SIFT()  # or cv2.SIFT() in OpenCV 2.4.9+
    #orb = SIFT(nfeatures = 5000)  # or cv2.SIFT() in OpenCV 2.4.9+

    # Find keypoints, compute descriptors and show them on original image (with scale and orientation)
    image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)
    #print "Image 1: {} keypoints found".format(len(image_1_kp))
    #print "Image 2: {} keypoints found".format(len(image_2_kp))

    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(image_1_desc, image_2_desc)
    #print "{} matches found".format(len(matches))

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x: x.distance)

    #num_matches = 10
    #print 'NUM MATCHES ---', num_matches
    return image_1_kp, image_2_kp, matches[:num_matches]
    # END OF FUNCTION.


def findHomography(image_1_kp, image_2_kp, matches):
    """ Returns the homography between the keypoints of image 1, image 2, and
        its matches.

    Follow these steps:
        1. Iterate through matches and:
            1a. Get the x, y location of the keypoint for each match. Look up
                the documentation for cv2.DMatch. Image 1 is your query image,
                and Image 2 is your train image. Therefore, to find the correct
                x, y location, you index into image_1_kp using match.queryIdx,
                and index into image_2_kp using match.trainIdx. The x, y point
                is stored in each keypoint (look up documentation).
            1b. Set the keypoint 'pt' to image_1_points and image_2_points, it
                should look similar to this inside your loop:
                    image_1_points[match_idx] = image_1_kp[match.queryIdx].pt
                    # Do the same for image_2 points.

        2. Call cv2.findHomography and pass in image_1_points, image_2_points,
           use method=cv2.RANSAC and ransacReprojThreshold=5.0. I recommend
           you look up the documentation on cv2.findHomography to better
           understand what these parameters mean.

        3. cv2.findHomography returns two values, the homography and a mask.
           Ignore the mask, and simply return the homography.

    Note:
        The unit test for this function in the included testing script may
        have value differences and thus may not pass. Please check your image
        results visually. If your output warped image looks fine, don't worry
        about this test too much.

    Args:
        image_1_kp (list): The image_1 keypoints, the elements are of type
                           cv2.KeyPoint.
        image_2_kp (list): The image_2 keypoints, the elements are of type
                           cv2.KeyPoint.
        matches (list): A list of matches. Each item in the list is of type
                        cv2.DMatch.
    Returns:
        homography (numpy.ndarray): A 3x3 homography matrix. Each item in
                                    the matrix is of type numpy.float64.
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    # WRITE YOUR CODE HERE.
    for match_idx, match in enumerate(matches):
        #print match_idx, match, type(match)
        #print match.queryIdx, match.trainIdx
        #print image_1_kp[match.queryIdx], image_2_kp[match.trainIdx]
        image_1_points[match_idx] = image_1_kp[match.queryIdx].pt
        image_2_points[match_idx] = image_2_kp[match.trainIdx].pt

    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, 5.0)
    #print type(homography), homography.shape, type(mask), mask.shape
    transform = homography
    # Replace this return statement with the homography.

    return transform
    # END OF FUNCTION


def getBoundingCorners(image_1, image_2, homography):
    """
    Find the coordinates of the top left corner and bottom right corner of the
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Follow these steps:
        1. Obtain the corners for image 1 and image 2 using the function you
        wrote above.

        2. Transform the perspective of the corners of image 1 by using
        the corners from step 1 and the homography to obtain the transformed
        corners.

        Note: Now we know where the corners of image 1 and image 2 will end up
        in the output image. Out of these 8 points (the transformed corners of
        image 1 and the corners of image 2), we want to find the minimum x,
        maximum x, minimum y, and maximum y. We will need this when warping the
        perspective of image 1.

        3. Join the two corner arrays together (the transformed image 1
        corners, and the image 2 corners) into one array of size (8, 1, 2).

        4. For the first column of this array, find the min and max. This will
        be your minimum and maximum X values. Store into x_min, x_max.

        5. For the second column of this array, find the min and max. This will
        be your minimum and maximum Y values. Store into y_min, y_max.

    Args:
        image_1 (numpy.ndarray): Left image.
        image_2 (numpy.ndarray): Right image.
        homography (numpy.ndarray): 3x3 matrix that represents the homography
                                    from image 1 to image 2.

    Returns:
        min_xy (numpy.ndarray): 2x1 array containing the coordinates of the
                                top left corner of the bounding rectangle
                                of a canvas large enough to fit both images
        max_xy (numpy.ndarray): 2x1 array containing the coordinates of the
                                bottom right corner of the bounding rectangle
                                of a canvas large enough to fit both images
    """
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    # WRITE YOUR CODE HERE - YOU ONLY NEED TO DEFINE THE FOUR VALUES:
    # x_min, y_min, x_max, y_max
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)

    # Used the sites for reference to transformed corners of image_1
    #http://www.learnopencv.com/homography-examples-using-opencv-python-c/
    #http://opencv.itseez.com/2.4/modules/core/doc/operations_on_arrays.html?highlight=perspectivetransform#perspectivetransform
    img1_trans = cv2.perspectiveTransform(corners_1, homography)
    cnct_crnrs = np.concatenate((img1_trans, corners_2))

    #FOR TESTING: cnct_crnrs = np.array([ [[1,2],[3,4],[5,6],[7,8]], [[9,10],[11,12],[13,14],[15,16]] ])
    #print img1_trans
    #print img1_trans.shape, corners_2.shape, cnct_crnrs.shape
    #print cnct_crnrs

    x_min = np.amin(cnct_crnrs[::, ::, :1:])
    y_min = np.amin(cnct_crnrs[::, ::, 1::])
    x_max = np.amax(cnct_crnrs[::, ::, :1:])
    y_max = np.amax(cnct_crnrs[::, ::, 1::])
    #print x_min, y_min, x_max, y_max
    #print '\n', cnct_crnrs[::, ::, :1:]

    '''
    multiply H by a homogeneous coordinate of image 1 to get the corresponding
    coordinate of image 2 (since Hb = wb', I had to divide each result by the third
    element w so that the third element would become 1 for proper homogeneous coordinates).

    print corners_1.shape, corners_1.dtype
    print '\n', homography, '\n'
    print '\n', corners_2

    ONE = np.ones((1, 1), dtype=np.float32)
    #for corner in corners_1:
    for corner_idx, corner in enumerate(corners_1):
        #print 'CORNER', corner
        corner = np.append(corner, ONE)
        #print 'ADD  1', corner, corner.dtype
        crd = np.matmul(corner.transpose(), homography)
        print 'CRDNT', crd[:2]
        #corners_2 = np.append(corners_2[corner_idx], crd[:2])
        print 'HERE ---', corners_2[corner_idx], corners_2[corner_idx].size, crd[:2].size
        #print np.append(corners_2, crd[:2])
    print corners_2.shape
    '''
    '''
    add_one = np.ones((1, 1), dtype=np.float32)
    print '\nSHAPE ---', add_one, add_one.shape
    #conv_image[::2,::2] = image[:,:]

    print '\n', homography
    print '\n', corners_1
    corners_1[::,::,::] = add_one
    #print '\nHERE ----', corners_1[::]
    for x in corners_1:
        print x


    print '\n\nTESTING------'
    x = np.array([ [[1,1],[1,2],[1,3],[1,10]], [[1,4],[1,5],[1,6],[11,1]] ])
    #x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    add_one = np.ones((x.shape[1], x.shape[0]))
    print add_one.shape
    print x.shape
    print x
    print '\n'
    x[::, ::, 1::] = 99
    print x
    for a in x:
        print a.shape, add_one.shape
        a[::, ::2] = add_one[::,::-1]
        print a
        break
    '''

    # END OF CODING
    min_xy = np.array([x_min, y_min])
    max_xy = np.array([x_max, y_max])
    return min_xy, max_xy
    # END OF FUNCTION


def warpCanvas(image_1, homography, min_xy, max_xy):
    """ Warps image 1 so it can be blended with image 2 (stitched).

    Follow these steps:
        1. Create a translation matrix that will shift the image by the
        required x_min and y_min (should be a numpy.ndarray). This looks
        like this:
            [[1, 0, -1 * x_min],
             [0, 1, -1 * y_min],
             [0, 0, 1]]

        Note: We'd like you to explain the reasoning behind multiplying the
        x_min and y_min by negative 1 in your writeup.

        2. Compute the dot product of your translation matrix and the
        homography in order to obtain the homography matrix with a translation.

        3. Then call cv2.warpPerspective. Pass in image 1, the dot product of
        the matrix computed in step 6 and the passed in homography and a vector
        that will fit both images, since you have the corners and their max and
        min, you can calculate it as (x_max - x_min, y_max - y_min), or use
        the size parameter below.

    Args:
        image_1 (numpy.ndarray): Left image.
        homography (numpy.ndarray): 3x3 matrix that represents the homography
                                    from image 1 to image 2.
        min_xy (numpy.ndarray): 2x1 array containing the coordinates of the
                                top left corner of a canvas large enough to
                                fit a warped image_1 and the next image from
                                the panorama.
        max_xy (numpy.ndarray): 2x1 array containing the coordinates of the
                                bottom right corner of a canvas large enough to
                                fit a warped image_1 and the next image from
                                the panorama.

    Returns:
        warped_image (numpy.ndarray): image_1 warped and inserted onto a canvas
                                      big enough to join with the next image
                                      in the panorama.
    """
    warped_image = None
    size = tuple(max_xy - min_xy)  # use in call to cv2.warpPerspective
    # WRITE YOUR CODE HERE
    trans_matrix = np.array([[1, 0, -1 * min_xy[0]],
                             [0, 1, -1 * min_xy[1]],
                             [0, 0, 1]])

    #print trans_matrix

    dot_p = np.dot(trans_matrix, homography)
    #print dot_p

    warped_image = cv2.warpPerspective(image_1, dot_p, size)
    # END OF CODING
    return warped_image
    # END OF FUNCTION


def blendImagePair(warped_image, image_2, point):
    """ This is the blending function. We provide a basic implementation of
    this function that we would like you to replace.

    This function takes in an image that has been warped and an image that
    needs to be inserted into the warped image. Lastly, it takes in a point
    where the new image will be inserted.

    The current method we provide is very simple, it pastes in the image at the
    point. We want you to replace this and blend between the images.

    We want you to be creative. The most common implementation would be to take
    the average between image 1 and image 2 only for the pixels that overlap.
    That is just a starting point / suggestion but you are encouraged to use
    other approaches.

    Args:
        warped_image (numpy.ndarray): The image from by cv2.warpPerspective.
        image_2 (numpy.ndarray): The image to insert into the warped image.
        point (numpy.ndarray): The point (x, y) to insert the image at.

    Returns:
        image: The warped image with image_2 blended into it.
    """
    output_image = np.copy(warped_image)
    # WRITE YOUR CODE HERE
    # REPLACE THIS WITH YOUR BLENDING CODE.

    #####################################
    # START Code from assignment 6, blend image:
    #####################################
    def generatingKernel(parameter):
        kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                           0.25, 0.25 - parameter /2.0])

        return np.outer(kernel, kernel)

    def reduce(image):
        kernel = generatingKernel(0.4)
        conv_image = scipy.signal.convolve2d(image, kernel, 'same')

        return conv_image[::2, ::2].astype(np.float64) 

    def expand(image):
        kernel = generatingKernel(0.4)
        conv_image = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)   #create np.zeros
        conv_image[::2,::2] = image[:,:]    #assign every other [row, col] the value of the image [row, col]
        return 4*scipy.signal.convolve2d(conv_image, kernel, 'same')

    def gaussPyramid(image, levels):
        output = [image]

        for i in range(levels):
            reduce_image = reduce(image)
            reduce_image = reduce_image.astype(np.float64)
            #print reduce_image.dtype
            output.append(reduce_image)
            image = reduce_image

        return output

    def laplPyramid(gaussPyr):
        output = []

        for gaussPyr_ind in range(0, len(gaussPyr) - 1, 1):
            gaussian = gaussPyr[gaussPyr_ind]
            expGaussPyr = expand(gaussPyr[gaussPyr_ind + 1])

            if gaussian.shape[0] != expGaussPyr.shape[0]:
                expGaussPyr = np.delete(expGaussPyr, -1, axis=0)

            if gaussian.shape[1] != expGaussPyr.shape[1]:
                expGaussPyr = np.delete(expGaussPyr, -1, axis=1)

            laplacian = gaussian - expGaussPyr
            output.append(laplacian)

        output.append(gaussPyr[-1])
        return output

    def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
        blendedPyr = []

        for i in range(0, len(laplPyrWhite), 1):
            #print 'HERE --', laplPyrWhite[i].shape, laplPyrBlack[i].shape, gaussPyrMask[i].shape
            output = np.zeros(shape=laplPyrWhite[i].shape, dtype=np.float64)

            output = ( gaussPyrMask[i] * laplPyrWhite[i] +
                       (1 - gaussPyrMask[i]) * laplPyrBlack[i] )

            blendedPyr.append(output)

        return blendedPyr

    def collapse(pyramid):

        for ind in range(len(pyramid), 0, -1):

            if ind - 2 >= 0:
                if ind - len(pyramid) == 0:  #first pass
                    pyr_exp = expand(pyramid[ind - 1])
                else:
                    pyr_exp = expand(sum_pyr)

                pyr = pyramid[ind - 2]

                if pyr_exp.shape[0] != pyr.shape[0]:
                    pyr_exp = np.delete(pyr_exp, -1, axis=0)
                if pyr_exp.shape[1] != pyr.shape[1]:
                    pyr_exp = np.delete(pyr_exp, -1, axis=1)

                sum_pyr = pyr_exp + pyr

        return sum_pyr

    def run_blend(black_image, white_image, mask):
        # Automatically figure out the size
        #print 'HERE ---', black_image.shape, white_image.shape, mask.shape
        min_size = min(black_image.shape)
        #depth = int(math.floor(math.log(min_size, 2))) - 4   #use math, at least 16x16 at the highest level.
        depth = int(np.floor(np.log2(min_size))) - 4       #use np, at least 16x16 at the highest level.
        #print 'LAYERS ---', depth
        
        gauss_pyr_mask = gaussPyramid(mask, depth)
        gauss_pyr_black = gaussPyramid(black_image, depth)
        gauss_pyr_white = gaussPyramid(white_image, depth)


        lapl_pyr_black  = laplPyramid(gauss_pyr_black)
        lapl_pyr_white = laplPyramid(gauss_pyr_white)

        outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
        outimg = collapse(outpyr)

        # Accountfor blending that sometimes results in slightly out of bound numbers
        outimg[outimg < 0] = 0
        outimg[outimg > 255] = 255
        outimg = outimg.astype(np.uint8)

        return outimg

    #####################################
    # END Code from assignment 6, blend image:
    #####################################

    # blend imagees
    RANGE = 10
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2

    mask_img = np.zeros(output_image.shape, dtype=np.uint8)
    mask_img[point[1]+RANGE:point[1] + image_2.shape[0]-RANGE,
             point[0]+RANGE:point[0] + image_2.shape[1]-RANGE] = 255

    #print output_image.dtype, mask_img.dtype, image_2.dtype, warped_image.dtype
    #print mask_img.shape, image_2.shape, warped_image.shape
    #mask_img[mask_img==0] = 255

    #cv2.imshow('IMAGE 2', mask_img)
    #cv2.waitKey()

    black_img = warped_image.astype(float)
    white_img = output_image.astype(float)
    mask_img = mask_img.astype(float) / 255

    out_layers = []

    '''
    #####################################
    # BEGIN TESTING to obtain border of inserted image
    #####################################
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2

    RANGE = 10
    #TOP Horizontal
    output_image[point[1]-RANGE:point[1]+RANGE,
                 point[0]-RANGE:point[0] + image_2.shape[1]+RANGE] = 0

    #BOTTOM Horizontal
    output_image[point[1] + image_2.shape[0]-RANGE:point[1] + image_2.shape[0]+RANGE,
                 point[0]-RANGE:point[0] + image_2.shape[1]+RANGE] = 255

    #LEFT Verticle
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]-RANGE:point[0]+RANGE] = 0

    #RIGHT Verticle
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0] + image_2.shape[1]-RANGE:point[0] + image_2.shape[1]+RANGE] = 255

    #cv2.imshow('IMAGE 2', output_image)
    #cv2.waitKey()
    cv2.imwrite('TEST_IMAGE.jpg', output_image)
    #####################################
    # BEGIN TESTING to obtain border of inserted image
    #####################################
    '''

    #print point[1], point[1] + image_2.shape[0]
    #print point[0], point[0] + image_2.shape[1]
    #print warped_image.shape, image_2.shape, output_image.shape



    for channel in range(3):
        #print 'CHAN ---', black_img.shape, white_img.shape, mask_img.shape
        outimg = run_blend(black_img[:,:,channel], white_img[:,:,channel], \
                           mask_img[:,:,channel])

        out_layers.append(outimg)
    outimg = cv2.merge(out_layers)
    output_image = outimg

    #print 'POINT ---', point
    #print '\nwarped_image', getImageCorners(warped_image)
    #print '\nimage_2', getImageCorners(image_2)

    #cv2.imshow('IMAGE W', warped_image)
    #cv2.imshow('IMAGE 2', image_2)
    #cv2.waitKey()

    '''
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2
    '''

    #print point[1], point[1] + image_2.shape[0]
    #print point[0], point[0] + image_2.shape[1]
    #print warped_image.shape, image_2.shape, output_image.shape

    return output_image
    # END OF FUNCTION


if __name__ == "__main__":
    # Some simple testing.
    image_1 = cv2.imread("images/1.jpg")
    image_2 = cv2.imread("images/2.jpg")
    image_1_kp, image_2_kp, matches = findMatchesBetweenImages(image_1,
                                                               image_2,
                                                               20)
    homography = findHomography(image_1_kp, image_2_kp, matches)
    min_xy, max_xy = getBoundingCorners(image_1, image_2, homography)
    blend_canvas = warpCanvas(image_1, homography, min_xy, max_xy)
    output_image = blendImagePair(blend_canvas, image_2, -1 * min_xy)
    cv2.imwrite("warped_image_1_2.jpg", output_image)
