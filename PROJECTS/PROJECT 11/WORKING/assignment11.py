# ASSIGNMENT 11
# Robert Bobkoskie

""" Assignment 11 - Video Textures

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

import numpy as np
import cv2
import scipy.signal


def videoVolume(images):
    """ Create a video volume from the image list.

    Note: Simple function to convert a list to a 4D numpy array, you should know
    how to do this.

    Args:
        images (list): A list of frames. Each element of the list contains a
                       numpy array of a colored image. You may assume that each
                       frame has the same shape, (rows, cols, 3).

    Returns:
        output (numpy.ndarray): A 4D numpy array. This array should have
                                dimensions (num_frames, rows, cols, 3) and
                                dtype np.uint8.
    """
    output = np.zeros((len(images), images[0].shape[0], images[0].shape[1],
                      images[0].shape[2]), dtype=np.uint8)

    # WRITE YOUR CODE HERE.
    #print output
    for i, img in enumerate(images):
        #print i, img
        output[i:i+1] = img
    #print output
    # END OF FUNCTION.
    return output

def computeSimilarityMetric(video_volume):
    """
    You need to compute the differences between each pair of frames in the
    video volume. The goal, of course, is to be able to tell how good
    a jump between any two frames might be so that the code you write later
    on can find the optimal loop. The closer the similarity metric is to zero,
    the more alike the two frames are.

    This is done by computing the square root of the sum of the differences
    between each frame in the video volume.  Then, we normalize the values
    into the range 0 to 1, to remove the resolution as a factor in the scores
    (This will help when you work with videos of different resolutions)

    Suggested Instructions:

        1. Create a for loop that goes through the video volume. Create a
           variable called cur_frame.

            A. Create another for loop that goes through the video volume
                again. Create a variable called comparison_frame.

                i. Inside this loop, compute this mathematical statement.
                    rssd = sum ( (cur_frame - comparison_frame)^2 ) ** 0.5

                ii. Set output[i, j] = ssd

        2.  Divide all the values in output by the average value.  This has
            two benefits: first, it removes any resolution dependencies: the
            same video at two different resolutions will end up with the same
            values.  Second, it distributes the values over a consistent range
            regardless of the video, so the rest of your code is not so exposed
            to the quirks of any given video.

    Hints:

        Remember the matrix is symmetrical, so when you are computing the
        similarity at i, j, its the same as computing the similarity at j, i so
        you don't have to do the math twice. This speeds up the function by 2.

        Also, the similarity at all i,i is always zero, no need to calculate it

    Args:
        video_volume (numpy.ndarray):
            A 4D numpy array with dimensions (num_frames, rows, cols, 3).

            This can be produced by the videoVolume function.

    Returns:
        output (numpy.ndarray):
            A square 2d numpy array of dtype np.float.

            output[i,j] should contain the similarity score from 0 to 1
            between all frames [i,j

            This matrix is symmetrical with a diagonal of zeros.

            The values should be of type np.float.
    """

    output = np.zeros((len(video_volume), len(video_volume)), dtype=np.float)

    # WRITE YOUR CODE HERE.
    '''
    [[ 0.          0.01251703  2.23934232]
    [ 0.01251703   0.          2.24814065]
    [ 2.23934232   2.24814065  0.]]
    Received
    [[ 0.          0.87708209  2.23612935]
    [ 0.87708209   0.          1.38678855]
    [ 2.23612935   1.38678855  0.]]
    '''
    #print video_volume
    otr_loop = []
    for i, this_frame in enumerate(video_volume):
        otr_loop.append(i)
        inr_loop = []
        for j, comp_frame in enumerate(video_volume):
            #print i, j
            if not j in otr_loop and not j in inr_loop:
                inr_loop.append(j)
                this_frame = this_frame.astype(np.float64)
                comp_frame = comp_frame.astype(np.float64)
                #print this_frame
                #print comp_frame
                #print 'DIFF ---\n', this_frame - comp_frame
                #ssd = np.sum(((this_frame[::] - comp_frame[::])**2))**0.5
                ssd = np.sum(((this_frame - comp_frame)**2))**0.5
                #print ssd
                output[i, j] = ssd
                output[j, i] = ssd

    #output = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #output = output.astype(np.float64)
    #print output
    #print np.average(output)
    output = output / np.average(output)
    #print output
    #print output.dtype
    # END OF FUNCTION.
    return output

def transitionDifference(ssd_difference):
    """ Compute the transition costs between frames, taking dynamics into
        account.

    Instructions:
        1. Iterate through the rows and columns of ssd difference, ignoring the
           first two values and the last two values.
            1a. For each value at i, j, multiply the binomial filter of length
                five by the weights starting two frames before until two frames
                after, and take the sum of those products.

                i.e. Your weights for frame i are:
                     [weight[i - 2, j - 2],
                      weight[i - 1, j - 1],
                      weight[i, j],
                      weight[i + 1, j + 1],
                      weight[i + 2, j + 2]]

                Multiply that by the binomial filter weights at each i, j to get
                your output.

                It may take a little bit of understanding to get why we are
                computing this, the simple explanation is that to change from
                frame 4 to 5, lets call this ch(4, 5), and we make this weight:

                ch(4, 5) = ch(2, 3) + ch(3, 4) + ch(4, 5) + ch(5, 6) + ch(6, 7)

                This accounts for the weights in previous changes and future
                changes when considering the current frame.

                Of course, we weigh all these sums by the binomial filter, so
                that the weight ch(4, 5) is still the most important one, but
                hopefully that gives you a better understanding.

    Args:
        ssd_difference (numpy.ndarray): A difference matrix as produced by your
                                        ssd function.

    Returns:
        output (numpy.ndarray): A difference matrix that takes preceding and
                                following frames into account. The output
                                difference matrix should have the same dtype as
                                the input, but be 4 rows and columns smaller,
                                corresponding to only the frames that have valid
                                dynamics.

    Hint: There is an efficient way to do this with 2d convolution. Think about
          the coordinates you are using as you consider the preceding and
          following frame pairings.
    """

    output = np.zeros((ssd_difference.shape[0] - 4,
                       ssd_difference.shape[1] - 4),
                      dtype=ssd_difference.dtype)
    # WRITE YOUR CODE HERE.
    #ssd_difference = np.arange(81).reshape(9,9) #FOR TESTING
    #print ssd_difference, '\n'
    #print ssd_difference[2:-2,2:-2]
    for i in range(2, ssd_difference.shape[0] - 2):
        for j in range(2, ssd_difference.shape[1] - 2):
            frame = ssd_difference[i-2:i+3,j-2:j+3]
            #next_frame = ssd_difference[i-1:i+2,j-1:j+2]

            #print i, j
            #print binomialFilter5()
            #print frame
            #print np.diag(frame)
            #print np.diag(frame)*binomialFilter5()

            '''
            #print scipy.signal.convolve2d(frame, next_frame, 'same')
            #conv_frame = scipy.signal.convolve2d(frame, next_frame, 'same')
            #print sum(conv_frame*binomialFilter5())
            '''

            #output[i-2, j-2] = sum(np.dot(binomialFilter5(), frame))
            output[i-2, j-2] = sum(np.diag(frame)*binomialFilter5())

    #print output
    # END OF FUNCTION.
    return output

def findBiggestLoop(transition_diff, alpha):
    """ Given the difference matrix, find the longest and smoothest loop that we
      can.

    Args:
        transition_diff (np.ndarray): A square 2d numpy array of dtype float.
                                      Each cell contains the cost of
                                      transitioning from frame i to frame j in
                                      the input video as returned by the
                                      transitionDifference function.

        alpha (float): a parameter for how heavily you should weigh the size of
                       the loop relative to the transition cost of the loop.
                       Larger alphas favor longer loops. Try really big values
                       to see what you get.

    start, end will be the indices in the transition_diff matrix that give the
    maximum score according to the following metric:
        score = alpha * (end - start) - transition_diff[end, start]

    Compute that score for every possible starting and ending index (within the
    size of the transition matrix) and find the largest score.

    See README.html for the scoring function to implement this function.

    Returns:
        start (int): The starting frame number of the longest loop.
        end (int): The final frame number of the longest loop.
    """
    start = 0
    end = 0
    largest_score = 0
    # WRITE YOUR CODE HERE.
    #print alpha
    for (i, j), diff in np.ndenumerate(transition_diff):
        #print i, j, diff
        score = alpha*(i - j) - transition_diff[i, j]
        if score > largest_score:
            largest_score = score
            start = j
            end = i

    # END OF FUNCTION.
    return start, end

def synthesizeLoop(video_volume, start, end):
    """ Pull out the given loop from the input video volume.

    Args:
        video_volume (np.ndarray): A (time, height, width, 3) array, as created
                                   by your videoVolume function.
        start (int): the index of the starting frame.
        end (int): the index of the ending frame.

    Returns:
        output (list): a list of arrays of size (height, width, 3) and dtype
                       np.uint8, similar to the original input the videoVolume
                       function.
    """

    output = []
    # WRITE YOUR CODE HERE.
    for i in range(start, end+1):
        #print i, video_volume[i]
        output.append(video_volume[i])

    # END OF FUNCTION.
    return output


def binomialFilter5():
    """ Return a binomial filter of length 5.

    Note: This is included for you to use.

    Returns:
        output (numpy.ndarray): A 5x1 numpy array representing a binomial
                                filter.
    """

    return np.array([1 / 16., 1 / 4., 3 / 8., 1 / 4., 1 / 16.],
                    dtype=float)


def runTexture(img_list, alpha):
    """ This function administrates the extraction of a video texture from the
    given frames.
    """
    video_volume = videoVolume(img_list)
    ssd_diff = computeSimilarityMetric(video_volume)
    transition_diff = transitionDifference(ssd_diff)
    # alpha = 1.0 / transition_diff.shape[0]

    print "Alpha is {}".format(alpha)
    idxs = findBiggestLoop(transition_diff, alpha)

    diff3 = np.zeros(transition_diff.shape, float)

    for i in range(transition_diff.shape[0]):
        for j in range(transition_diff.shape[1]):
            diff3[i, j] = alpha * (i - j) - transition_diff[i, j]

    return (vizDifference(ssd_diff),
            vizDifference(transition_diff),
            vizDifference(diff3),
            synthesizeLoop(video_volume, idxs[0] + 2, idxs[1] + 2))


def vizDifference(diff):
    return (((diff - diff.min()) /
             (diff.max() - diff.min())) * 255).astype(np.uint8)


def readImages(image_dir):
    """ This function reads in input images from a image directory

    Note: This is implemented for you since its not really relevant to
    computational photography (+ time constraints).

    Args:
        image_dir (str): The image directory to get images from.

    Returns:
        images(list): List of images in image_dir. Each image in the list is of
                      type numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = reduce(list.__add__, map(glob, search_paths))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
              for f in image_files]

    return images


if __name__ == "__main__":
    import errno
    import os
    import sys

    from glob import glob

    alpha = 1 if len(sys.argv) < 2 else float(sys.argv[-1])
    video_dir = "candle"
    image_dir = os.path.join("videos", "source", video_dir)
    out_dir = os.path.join("videos", "out")

    print "Reading images."
    images = readImages(image_dir)

    print "Computing video texture..."
    diff1, diff2, diff3, out_list = runTexture(images, alpha)

    try:
        os.makedirs(os.path.join(out_dir, video_dir))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    cv2.imwrite(os.path.join(out_dir,
                '{}diff1.png'.format(video_dir)),
                diff1)
    cv2.imwrite(os.path.join(out_dir,
                '{}diff2.png'.format(video_dir)),
                diff2)
    cv2.imwrite(os.path.join(out_dir,
                '{}diff3.png'.format(video_dir)),
                diff3)

    for idx, image in enumerate(out_list):
        cv2.imwrite(os.path.join(out_dir, video_dir,
                    'frame{0:04d}.png'.format(idx)), image)
