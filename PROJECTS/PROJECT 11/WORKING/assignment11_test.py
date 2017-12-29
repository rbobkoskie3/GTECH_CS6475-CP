import cv2
import numpy as np
import sys
import os

import assignment11


def test_videoVolume():
    image_list = [np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.uint8),
                  np.array([[[1, 1, 1], [1, 1, 1]]], dtype=np.uint8),
                  np.array([[[2, 2, 2], [2, 2, 2]]], dtype=np.uint8),
                  np.array([[[3, 3, 3], [3, 3, 3]]], dtype=np.uint8)]

    ans = np.ndarray([4, 1, 2, 3], dtype=np.uint8)

    solution = assignment11.videoVolume(image_list)

    type_and_shape_test(solution, ans, "videoVolume")

    return True


def test_computeSimilarity():

    video_volume = np.array([[[[0, 0, 0], [0, 0, 0]]],
                             [[[1, 1, 1], [1, 1, 1]]],
                             [[[2, 2, 2], [2, 2, 2]]],
                             [[[3, 3, 3], [3, 3, 3]]]], dtype=np.uint8)

    ans = np.ndarray([4, 4], dtype=np.float)

    solution = assignment11.computeSimilarityMetric(video_volume)

    type_and_shape_test(solution, ans, "computeSimilarity")

    return True


def test_transitionDifference():

    ssd = np.zeros((9, 9), dtype=float)
    ssd[4, 4] = 1

    ans = np.ndarray([5, 5], dtype=float)

    solution = assignment11.transitionDifference(ssd)

    type_and_shape_test(solution, ans, "transitionDifference")

    return True


def test_findBiggestLoop():
    diff = np.ones((5, 5), dtype=float)
    alpha = 1

    print "Evaluating findBiggestLoop."
    solution = assignment11.findBiggestLoop(diff, alpha)

    if type(solution) is not tuple:
        print ("Error: findBiggestLoop has type {}. " +
               "Expected type is {}.").format(type(solution),
                                              tuple)
        return False

    print "findBiggestLoop passed.\n"
    return True


def test_synthesizeLoop():
    video_volume = np.array([[[[0, 0, 0],
                               [0, 0, 0]]],
                             [[[1, 1, 1],
                               [1, 1, 1]]],
                             [[[2, 2, 2],
                               [2, 2, 2]]],
                             [[[3, 3, 3],
                               [3, 3, 3]]]], dtype=np.uint8)
    frames = (2, 3)

    print "Evaluating synthesizeLoop."
    solution = assignment11.synthesizeLoop(video_volume,
                                           frames[0], frames[1])

    if type(solution) is not list:
        print ("Error: synthesizeLoop has type {}. " +
               "Expected type is {}.").format(type(solution),
                                              type(list))
        return False

    if len(solution) != 2:
        print ("Error: synthesizeLoop has len {}. " + 
               "Expected len is {}.").format(len(solution), 2)
        return False

    print "synthesizeLoop passed.\n"
    return True


def type_and_shape_test(solution, ans, function_name):

    print "Evaluating {}.".format(function_name)

    # Test for type
    if type(solution) is not type(ans):
      raise ValueError("Error - {} output type is {}. " +
                  "Expected type is {}.").format(function_name,
                                                 type(solution),
                                                 type(ans))
    # Test for shape
    if solution.shape != ans.shape:
      raise ValueError(("Error - {} output shape is {}. " +
                     "Expected shape is {}.").format(function_name,
                                                     solution.shape,
                                                     ans.shape))

    # Test for type of values in matrix
    if solution.dtype.type is not ans.dtype.type:
      raise ValueError(("Error - {} output values have type {}. " +
                  "Expected type is {}.").format(function_name,
                                                 solution.dtype.type,
                                                 ans.dtype.type))

    print "type_and_shape_test passed for {}.\n".format(function_name)


if __name__ == "__main__":
    print "Performing Unit Tests\n"
    if not test_videoVolume():
        print "videoVolume function failed. Halting testing."
        sys.exit()
    if not test_computeSimilarity():
        print "sumSquaredDifferences function failed. Halting testing."
        sys.exit()
    if not test_transitionDifference():
        print "transitionDifference function failed. Halting testing."
        sys.exit()
    if not test_findBiggestLoop():
        print "findBiggestLoop function failed. Halting testing."
        sys.exit()
    if not test_synthesizeLoop():
        print "synthesizeLoop function failed. Halting testing."
        sys.exit()

    print 'Unit tests passed.\n'

    sourcefolder = os.path.abspath(os.path.join(os.curdir, 'videos', 'source'))
    outfolder = os.path.abspath(os.path.join(os.curdir, 'videos', 'out'))
    # Ensure that the directory that holds our output directories exists...
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    print "Generating test output. (This could take a while.)"
    print 'Searching for video folders in {} folder'.format(sourcefolder)

    # Extensions recognized by opencv
    exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg',
            '.jpe', '.jp2', '.tiff', '.tif', '.png']

    alpha = 1 if len(sys.argv) < 2 else float(sys.argv[-1])

    # For every image in the source directory
    for video_dir in os.listdir(sourcefolder):
        print "Collecting images from directory {}".format(video_dir)
        img_list = []
        filenames = sorted(os.listdir(os.path.join(sourcefolder, video_dir)))

        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext in exts:
                img_list.append(cv2.imread(os.path.join(sourcefolder,
                                                        video_dir,
                                                        filename)))

        print "Extracting video texture frames."
        diff1, diff2, diff3, out_list = assignment11.runTexture(img_list,
                                                                alpha)

        cv2.imwrite(os.path.join(outfolder, '{}diff1.png'.format(video_dir)),
                    diff1)
        cv2.imwrite(os.path.join(outfolder, '{}diff2.png'.format(video_dir)),
                    diff2)
        cv2.imwrite(os.path.join(outfolder, '{}diff3.png'.format(video_dir)),
                    diff3)

        print "writing output to {}".format(os.path.join(outfolder, video_dir))
        if not os.path.exists(os.path.join(outfolder, video_dir)):
            os.mkdir(os.path.join(outfolder, video_dir))

        for idx, image in enumerate(out_list):
            cv2.imwrite(os.path.join(outfolder, video_dir,
                        'frame{0:04d}.png'.format(idx)), image)
