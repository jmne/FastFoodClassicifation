import os
from os import listdir
import matplotlib.image as mpimg
import numpy as np


def read_images(read_from_processed=False):
    test_images = []
    train_images = []
    valid_images = []
    invalid = 0

    if os.path.isdir('../resources/processed') and read_from_processed:
        test_images = np.load("../resources/processed/test_images.npy", allow_pickle=True)
        train_images = np.load("../resources/processed/train_images.npy", allow_pickle=True)
        valid_images = np.load("../resources/processed/valid_images.npy", allow_pickle=True)
    else:
        folder_dir = "../resources"
        for _dir in os.listdir(folder_dir):
            if _dir == "processed":
                continue
            for folder in listdir(folder_dir + "/" + _dir):
                for file in listdir(folder_dir + "/" + _dir + "/" + folder):
                    if (folder_dir + "/" + _dir + "/" + folder + "/" + file).endswith(".jpeg"):
                        if _dir == "test":
                            test_images.append(mpimg.imread(folder_dir + "/" + _dir + "/" + folder + "/" + file))
                        if _dir == "train":
                            train_images.append(mpimg.imread(folder_dir + "/" + _dir + "/" + folder + "/" + file))
                        if _dir == "valid":
                            valid_images.append(mpimg.imread(folder_dir + "/" + _dir + "/" + folder + "/" + file))
                    else:
                        invalid += 1
        print("Invalid images: ", invalid)

        np.save("../resources/processed/test_images.npy", np.asarray(test_images, dtype=object))
        np.save("../resources/processed/train_images.npy", np.asarray(train_images, dtype=object))
        np.save("../resources/processed/valid_images.npy", np.asarray(valid_images, dtype=object))

    return test_images, train_images, valid_images
