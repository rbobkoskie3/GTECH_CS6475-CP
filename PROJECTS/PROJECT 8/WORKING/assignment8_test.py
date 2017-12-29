
import sys
import os
import numpy as np
import cv2

import assignment8


def test_getImageCorners():
    """ This script will perform a unit test of the getImageCorners function
    """
    matrix = np.zeros((320, 320, 3))
    ans = np.ndarray((4, 1, 2), dtype=np.float32)

    print "\nEvaluating getImageCorners."

    corners = assignment8.getImageCorners(matrix)

    # Test for type.
    if not type(corners) == type(ans):
        raise TypeError(
            ("Error - corners has type {}." +
             " Expected type is {}.").format(type(corners), type(ans)))

    # Test for shape.
    if not corners.shape == ans.shape:
        raise ValueError(
            ("Error - corners has shape {}." +
             " Expected shape is {}.").format(corners.shape, ans.shape))

    # Test for type of values in matrix.
    if not type(corners.dtype.type) == type(ans.dtype.type):
        raise TypeError(
            ("Error - corners values have type {}." +
             "Expected type is {}.").format(type(corners[0][0][0]),
                                            type(ans[0][0][0])))
    print "getImageCorners tests passed."
    return True


def test_findMatchesBetweenImages():
    """ This script will perform a unit test on the matching function.
    """
    # Hard code output matches.
    image_1 = cv2.imread("images/source/panorama_1/1.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")

    print "\nEvaluating findMatchesBetweenImages."

    image_1_kp, image_2_kp, matches = \
        assignment8.findMatchesBetweenImages(image_1, image_2, 20)

    if not type(image_1_kp) == list:
        raise TypeError(
            "Error - image_1_kp has type {}. Expected type is {}.".format(
                type(image_1_kp), list))

    if len(image_1_kp) > 0 and \
        not type(image_1_kp[0]) == type(cv2.KeyPoint()):
        raise TypeError(("Error - The items in image_1_kp have type {}. " + \
                         "Expected type is {}.").format(type(image_1_kp[0]),
                                                        type(cv2.KeyPoint())))

    if not type(image_2_kp) == list:
        raise TypeError(
            "Error - image_2_kp has type {}. Expected type is {}.".format(
                type(image_2_kp), list))

    if len(image_2_kp) > 0 and \
        not type(image_2_kp[0]) == type(cv2.KeyPoint()):
        raise TypeError(("Error - The items in image_2_kp have type {}. " + \
                         "Expected type is {}.").format(type(image_2_kp[0]),
                                                        type(cv2.KeyPoint())))

    if not type(matches) == list:
        raise TypeError(
            "Error - matches has type {}. Expected type is {}. ".format(
                type(matches), list))

    if len(matches) > 0 and not type(matches[0]) == type(cv2.DMatch()):
        raise TypeError(("Error - The items in matches have type {}. " + \
                         "Expected type is {}.").format(type(matches[0]),
                                                        type(cv2.DMatch())))

    print "findMatchesBetweenImages testing passed.\n"
    return True


def test_findHomography():
    """ This function performs a unit test of the findHomography function
    """

    image_1 = cv2.imread("images/source/panorama_1/1.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")

    image_1_kp, image_2_kp, matches = assignment8.findMatchesBetweenImages(
        image_1, image_2, 20)

    homography = assignment8.findHomography(image_1_kp, image_2_kp, matches)

    ans = np.ndarray((3, 3), dtype=np.float64)

    print "Evaluating findHomography."

    # Test for type.
    if not type(homography) == type(ans):
        raise TypeError(
            ("Error - homography has type {}. " +
             "Expected type is {}.").format(type(homography), type(ans)))

    # Test for shape.
    if not homography.shape == ans.shape:
        raise ValueError(
            ("Error - homography has shape {}." +
             " Expected shape is {}.").format(homography.shape, ans.shape))

    # test for value type
    if not type(homography.dtype.type) == type(ans.dtype.type):
        raise TypeError(
            ("Error - The items in homography have type {}. " + 
             "Expected type is {}.").format(homography.dtype.type,
                                            ans.dtype.type))

    print "findHomography testing passed.\n"
    return True


def test_getBoundingCorners():
    """ Perform a unit test of getBoundingCorners function
    """
    # Hard code output matches.
    image_1 = cv2.imread("images/source/panorama_1/1.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")

    homography = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]],
                          dtype=np.float64)

    corners, _ = assignment8.getBoundingCorners(image_1, image_2, homography)

    ans = np.ndarray((2), dtype=np.float64)

    print "Evaluating getBoundingCorners."

    # Test for type.
    if not type(corners) == type(ans):
        raise TypeError(
            ("Error - getBoundingCorners has type {}. " +
             "Expected type is {}.").format(type(corners), type(ans)))

    # Test for shape.
    if not corners.shape == ans.shape:
        raise ValueError(
            ("Error - getBoundingCorners has shape {}." +
             " Expected shape is {}.").format(corners.shape, ans.shape))

    # Test for value type.
    if not type(corners.dtype.type) == type(ans.dtype.type):
        raise TypeError(
            ("Error - The items in getBoundingCorners have type {}. " + 
             "Expected type is {}.").format(corners.dtype.type,
                                            ans.dtype.type))

    print "getBoundingCorners testing passed.\n"
    return True


def test_warpCanvas():
    """ Perform a unit test of warpCanvas function
    """

    # Hard code output matches.
    image_1 = cv2.imread("images/source/panorama_1/1.jpg")

    homography = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]],
                          dtype=np.float64)

    min_xy = np.array([0., 0.], dtype=np.float32)
    max_xy = np.array(image_1.shape[:2][::-1], dtype=np.float32)

    warped_img = assignment8.warpCanvas(image_1, homography,
                                        min_xy, max_xy)

    ans = np.ndarray(image_1.shape, dtype=np.uint8)

    print "Evaluating getBoundingCorners."

    # Test for type.
    if not type(warped_img) == type(ans):
        raise TypeError(
            ("Error - warpCanvas output has type {}. " +
             "Expected type is {}.").format(type(warped_img), type(ans)))

    # Test for shape.
    if not warped_img.shape == ans.shape:
        raise ValueError(
            ("Error - warpCanvas output has shape {}." +
             " Expected shape is {}.").format(warped_img.shape, ans.shape))

    # Test for value type.
    if not type(warped_img.dtype.type) == type(ans.dtype.type):
        raise TypeError(
            ("Error - The items in warpCanvas output have type {}. " +
             "Expected type is {}.").format(warped_img.dtype.type,
                                            ans.dtype.type))

    print "warpedCanvas testing passed.\n"
    return True



def test_blendImagePair():
    warped_image = cv2.imread("images/testing/warped_image.jpg")
    image_2 = cv2.imread("images/source/panorama_1/2.jpg")
    point = (1107.26, 506.64)

    blended = assignment8.blendImagePair(warped_image, image_2, point)

    type_answer = np.copy(warped_image)
    type_answer[np.int(point[1]):np.int(point[1]) + image_2.shape[0],
                np.int(point[0]):np.int(point[0]) + image_2.shape[1]] = image_2

    print "Evaluating blendImagePair"

    # Test for type.
    if not type(blended) == type(type_answer):
        raise TypeError(
            ("Error - blended_image has type {}. " +
             "Expected type is {}.").format(type(blended), type(type_answer)))

    # Test for shape.
    if not blended.shape == type_answer.shape:
        raise ValueError(
            ("Error - blended_image has shape {}. " +
             "Expected shape is {}.").format(blended.shape, type_answer.shape))

    # Check if output is equivalent.
    if np.array_equal(blended, type_answer):
        print "WARNING: Blended image function has not been changed. You " + \
              "need to add your own functionality or you will not get " + \
              "credit for its implementation."

    print "blendImagePair testing passed.\n"
    return True


if __name__ == "__main__":
    print "Performing unit test."
    if not test_getImageCorners():
        print "getImageCorners function failed. Halting testing."
        sys.exit()
    if not test_findMatchesBetweenImages():
        print "findMatchesBetweenImages function failed. Halting testing."
        sys.exit()
    if not test_findHomography():
        print "findHomography function failed. Halting testing."
        sys.exit()
    if not test_getBoundingCorners():
        print "getBoundingCorners function failed. Halting testing."
        sys.exit()
    if not test_warpCanvas():
        print "warpCanvas function failed. Halting testing."
        sys.exit()
    if not test_blendImagePair():
        print "blendImagePair function failed. Halting testing."
        sys.exit()
    print "Unit test passed."
    print ("NOTE: THESE TESTS ONLY VERIFY TYPE, SHAPE, AND VALUE TYPE FROM " +
           "YOUR FUNCTIONS; THE AUTOGRADER ALSO TESTS FUNCTIONALITY.")

    sourcefolder = os.path.abspath(os.path.join(os.curdir, "images", "source"))
    outfolder = os.path.abspath(os.path.join(os.curdir, "images", "output"))

    print "Image source folder: {}".format(sourcefolder)
    print "Image output folder: {}".format(outfolder)

    print "Searching for folders with images in {}.".format(sourcefolder)

    # Extensions recognized by opencv
    exts = [".bmp", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".jpeg", ".jpg",
            ".jpe", ".jp2", ".tiff", ".tif", ".png"]

    # For every image in the source directory
    for dirname, dirnames, filenames in os.walk(sourcefolder):
        setname = os.path.split(dirname)[1]

        panorama_inputs = []
        panorama_filepaths = []

        for filename in filenames:
            print 'FILENAME', filename
            name, ext = os.path.splitext(filename)
            if ext.lower() in exts:
                panorama_filepaths.append(os.path.join(dirname, filename))
        panorama_filepaths.sort()

        for pan_fp in panorama_filepaths:
            panorama_inputs.append(cv2.imread(pan_fp))

        if len(panorama_inputs) > 1:
            print ("Found {} images in folder {}. " + \
                   "Processing them.").format(len(panorama_inputs), dirname)
        else:
            continue

        cur_img = panorama_inputs[0]
        for new_img in panorama_inputs[1:]:
            print "Computing matches."
            image_1_kp, image_2_kp, matches = \
                assignment8.findMatchesBetweenImages(cur_img, new_img, 20)
            print "Computing homography."
            homography = assignment8.findHomography(image_1_kp, image_2_kp,
                                                    matches)
            print "Finding the boundaries of the combined image."
            min_xy, max_xy = assignment8.getBoundingCorners(cur_img, new_img,
                                                            homography)
            print "Warping the image canvas to make room for the new image."
            cur_img = assignment8.warpCanvas(cur_img, homography,
                                             min_xy, max_xy)
            print "Blending the images."
            cur_img = assignment8.blendImagePair(cur_img, new_img, -min_xy)

        print "Writing output image to {}".format(outfolder) 
        cv2.imwrite(os.path.join(outfolder, setname) + ".jpg", cur_img)
