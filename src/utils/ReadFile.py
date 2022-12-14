import os
from os import listdir
import matplotlib.image as mpimg
import numpy as np


def read_images():
    """Read folder with images

    Returns:
        array: array with folder names
    """

    test_images = []

    # get the path/directory
    folder_dir = "../resources/Test"
    for folder in listdir(folder_dir):
        print(folder)
        for file in listdir(folder_dir + "/" + folder):
            test_images.append(mpimg.imread(folder_dir + "/" + folder + "/" + file))
    print(test_images)
    return test_images


        