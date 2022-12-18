import os
from os import listdir
from inspect import getsourcefile
from os.path import abspath
import matplotlib.image as mpimg
import numpy as np


def read_images(read_from_processed=False):
    """
    Read images from folder

    Args:
        read_from_processed: write your description
    """
    test_images = []
    train_images = []
    valid_images = []
    y_test = []
    y_train = []
    y_valid = []
    invalid = 0

    # Find src directory (assumes that this file is within src directory)
    src_dir = abspath(getsourcefile(lambda:0))
    while not src_dir.endswith("src"):
        src_dir = src_dir[:src_dir.rfind("\\")]
    os.chdir(src_dir)

    if os.path.isdir('../resources/processed') and read_from_processed:
        test_images = np.load("../resources/processed/test_images.npy", allow_pickle=True)
        train_images = np.load("../resources/processed/train_images.npy", allow_pickle=True)
        valid_images = np.load("../resources/processed/valid_images.npy", allow_pickle=True)
        y_test = np.load("../resources/processed/y_test.npy", allow_pickle=True)
        y_train = np.load("../resources/processed/y_train.npy", allow_pickle=True)
        y_valid = np.load("../resources/processed/y_valid.npy", allow_pickle=True)
    else:
        folder_dir = "../resources"
        for _dir in os.listdir(folder_dir):
            if _dir == "processed":
                continue
            for folder in listdir(folder_dir + "/" + _dir):
                print(folder)
                label = folder
                for file in listdir(folder_dir + "/" + _dir + "/" + folder):
                    if (folder_dir + "/" + _dir + "/" + folder + "/" + file).endswith(".jpeg" or ".jpg" or ".png"):
                        if _dir == "test":
                            test_images.append(mpimg.imread(folder_dir + "/" + _dir + "/" + folder + "/" + file))
                            y_test.append(label)
                        if _dir == "train":
                            train_images.append(mpimg.imread(folder_dir + "/" + _dir + "/" + folder + "/" + file))
                            y_train.append(label)
                        if _dir == "valid":
                            valid_images.append(mpimg.imread(folder_dir + "/" + _dir + "/" + folder + "/" + file))
                            y_valid.append(label)
                    else:
                        invalid += 1
                        os.remove(folder_dir + "/" + _dir + "/" + folder + "/" + file)
        print("Invalid images: ", invalid)

        if not os.path.isdir('../resources/processed'):
            os.mkdir('../resources/processed')
        
        np.save("../resources/processed/test_images.npy", np.asarray(test_images, dtype=object))
        np.save("../resources/processed/train_images.npy", np.asarray(train_images, dtype=object))
        np.save("../resources/processed/valid_images.npy", np.asarray(valid_images, dtype=object))
        np.save("../resources/processed/y_test.npy", np.asarray(y_test, dtype=str))
        np.save("../resources/processed/y_train.npy", np.asarray(y_train, dtype=str))
        np.save("../resources/processed/y_valid.npy", np.asarray(y_valid, dtype=str))

    return test_images, train_images, valid_images, y_test, y_train, y_valid
