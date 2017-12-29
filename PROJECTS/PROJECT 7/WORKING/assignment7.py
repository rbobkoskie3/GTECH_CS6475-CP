import numpy as np
import scipy as sp
import scipy.signal
import cv2

# Import ORB as SIFT to avoid confusion.
try:
  from cv2 import ORB as SIFT
except ImportError:
  try:
    from cv2 import SIFT
  except ImportError:
    try:
      SIFT = cv2.ORB_create
    except:
      raise AttributeError("Version of OpenCV(%s) does not have SIFT / ORB."
                      % cv2.__version__)

# This magic line appears to be needed with OpenCV3 to prevent the feature
# detector from throwing an error...
cv2.ocl.setUseOpenCL(False)

""" Assignment 7 - Feature Detection and Matching

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

def findMatchesBetweenImages(image_1, image_2):
  """ Return the top 10 list of matches between two input images.

  This function detects and computes SIFT (or ORB) from the input images, and
  returns the best matches using the normalized Hamming Distance.

  Follow these steps:
  1. Compute SIFT keypoints and descriptors for both images
  2. Create a Brute Force Matcher, using the hamming distance (and set
     crossCheck to true).
  3. Compute the matches between both images.
  4. Sort the matches based on distance so you get the best matches.
  5. Return the image_1 keypoints, image_2 keypoints, and the top 10 matches in
     a list.

  Note: We encourage you use OpenCV functionality (also shown in lecture) to
  complete this function.

  Args:
    image_1 (numpy.ndarray): The first image (grayscale).
    image_2 (numpy.ndarray): The second image. (grayscale).

  Returns:
    image_1_kp (list): The image_1 keypoints, the elements are of type
                       cv2.KeyPoint.
    image_2_kp (list): The image_2 keypoints, the elements are of type 
                       cv2.KeyPoint.
    matches (list): A list of matches, length 10. Each item in the list is of
                    type cv2.DMatch.

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

  # WRITE YOUR CODE HERE.
  # Code modified from:
  # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
  # Code also pulled from 'feature_matching.zip'
  # print 'DTYPE', image_1.dtype

  # Initialize ORB detector object
  orb = SIFT()  # or cv2.SIFT() in OpenCV 2.4.9+

  # Find keypoints, compute descriptors and show them on original image (with scale and orientation)
  image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
  image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)
  #print "Image 1: {} keypoints found".format(len(kp1))
  #print "Image 2: {} keypoints found".format(len(kp2))

  # Create BFMatcher (Brute Force Matcher) object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Match descriptors
  matches = bf.match(image_1_desc, image_2_desc)
  print "{} matches found".format(len(matches))

  # Sort them in the order of their distance
  matches = sorted(matches, key = lambda x: x.distance)
  matches = matches[:10]
  # Draw first 10 matches
  #img2 = np.zeros((1,1))
  #img_out = cv2.drawMatches(image_1, image_1_kp, image_2, image_2_kp, matches[:10], img2, flags=2)

  #cv2.imshow('IMAGE', img_out)
  #cv2.waitKey()
  #cv2.imwrite('IMAGE.jpg', img_out)

  '''
  sift = SIFT()

  # find the keypoints and descriptors with SIFT
  image_1_kp, image_1_desc = sift.detectAndCompute(image_1, None)
  image_2_kp, image_2_desc = sift.detectAndCompute(image_2, None)

  # BFMatcher with default params
  #bf = cv2.BFMatcher()
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.knnMatch(image_1_desc, image_2_desc, k=2)
  print type(matches)

  print "{} matches found".format(len(matches))
  # Sort them in the order of their distance
  #matches = sorted(matches, key = lambda x: x.distance)

  #img2 = np.zeros((1,1))
  #img_out = cv2.drawMatches(image_1, image_1_kp, image_2, image_2_kp, matches[:10], img2, flags=2)

  #img = np.zeros(shape=image_1.shape, dtype=np.float64)
  #img_out = cv2.drawMatches(image_1, image_1_kp, image_2, image_2_kp, matches[:10], img, flags=2)
  #output = cv2.drawMatches(image_1, image_1_kp, image_2, image_2_kp, matches)
  #img3 = cv2.drawMatchesKnn(image_1, image_1_kp, image_2, image_2_kp, good, flags=2)

  # We coded the return statement for you. You are free to modify it -- just
  # make sure the tests pass.
  '''
  return image_1_kp, image_2_kp, matches
  # END OF FUNCTION.
