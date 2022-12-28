import os

from matplotlib import pyplot as plt
from models import Model1, Model2, Model3
from utils import ReadFile
import time
from random import randrange
import sys

import numpy as np
import math
from PIL import Image

def resize(images, width, height):
    # Initialize new np array for resized images
    new_images = np.empty((images.shape[0], height, width, 3), dtype=np.uint8) # Assumes RGB format
    for index, image in enumerate(images):
        # Convert np array to PIL image (for resizing / cropping)
        image = Image.fromarray(image)
        # Enlarge/compress image so that the bottleneck dimension matches the desired dimension
        width_ratio = width / image.width
        height_ratio = height / image.height
        if (height_ratio > width_ratio):
            new_height = height
            new_width = round(image.width * height_ratio)
            image = image.resize((new_width, new_height))
        else:
            new_height = round(image.height * width_ratio)
            new_width = width
            image = image.resize((new_width, new_height))
        # Crop image to desired size
        left = (image.width - width) // 2
        upper = (image.height - height) // 2
        right = left + width
        lower = upper + height
        image = image.crop((left, upper, right, lower))
        new_images[index] = np.asarray(image)

    return new_images

def zca(images, batch_size=100):
    # Step 1
    images = images.reshape((images.shape[0], images.shape[1]*images.shape[2]*images.shape[3]))
    zca_images = np.empty(images.shape)
    num_batches = math.ceil(images.shape[0] / batch_size)
    for i in range(num_batches):
        start_idx = batch_size * i
        end_idx = min(batch_size * (i+1), images.shape[0])
        batch = images[start_idx : end_idx]
        print("Filtering images", start_idx + 1, "through", end_idx)
        # Step 2
        batch_norm = batch / 255
        # Step 3
        batch_norm = batch_norm - batch_norm.mean(axis=0)
        # Step 4
        cov = np.cov(batch_norm, rowvar=False)
        # Step 5
        U,S,V = np.linalg.svd(cov)
        # Step 6
        epsilon = 0.1
        # images_zca = U.dot(np.diag(1.0/np.sqrt(np.diag(S) + epsilon))).dot(U.T).dot(images_norm.T).T # Mine
        batch_zca = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(batch_norm.T).T # Tutorial
        # And rescale
        batch_zca_rescaled = (batch_zca - batch_zca.min()) / (batch_zca.max() - batch_zca.min())
        zca_images[start_idx : end_idx] = batch_zca_rescaled
    return zca_images;


def main():
    """
    Example application that processes a set of images.

    Args:
    """
    print("Start model..")
    if not os.path.isdir('../logs'):
        os.mkdir('../logs')
    if not os.path.isdir('../logs/plots'):
        os.mkdir('../logs/plots')
    open('../logs/log.txt', 'w').close()
    sys.stdout = open('../logs/log.txt', 'a')
    start = time.time()

    # test_images, train_images, valid_images, y_test, y_train, y_valid = ReadFile.read_images(
    #     read_from_processed=False)  # you need to process them once before!
    # print("Train images: ", len(train_images))
    # print("Valid images: ", len(valid_images))
    # print("Test images: ", len(test_images))
    # plt.figure(figsize=(10, 10))
    # for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     rand = randrange(len(train_images)-1)
    #     plt.imshow(train_images[rand])
    #     plt.axis("off")
    #     plt.title(y_train[rand])
    # plt.show()
    #
    print("Time elapsed: ", round(time.time() - start), "s", file=sys.stderr)
    print("Starting model...", file=sys.stderr)

    # Model1.model()
    # Model2.model()
    Model3.model()

    print("Compute Time: ", round(time.time() - start), "s", file=sys.stderr)

    sys.stdout.close()

    print("Done!", file=sys.stderr)

    exit(0)


if __name__ == '__main__':
    main()
