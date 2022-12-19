from matplotlib import pyplot as plt
from models import Model1
from utils import ReadFile
import time

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

def zca(images):
    # Step 1
    images = images.reshape((images.shape[0], width*height*rgb))
    # Step 2
    images_norm = images / 255
    # Step 3
    images_norm = images_norm - images_norm.mean(axis=0)
    # Step 4
    cov = np.cov(images_norm, rowvar=False)
    # Step 5
    U,S,V = np.linalg.svd(cov)
    # Step 6
    epsilon = 0.1
    # images_zca = U.dot(np.diag(1.0/np.sqrt(np.diag(S) + epsilon))).dot(U.T).dot(images_norm.T).T # Mine
    images_zca = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(images_norm.T).T # Tutorial
    # And rescale
    images_zca_rescaled = (images_zca - images_zca.min()) / (images_zca.max() - images_zca.min())
    return images_zca_rescaled;


def main():
    """
    Example application that processes a set of images.

    Args:
    """
    start = time.time()
    print("Starting reading images")
    test_images, train_images, valid_images, y_test, y_train, y_valid = ReadFile.read_images(
        read_from_processed=True)  # you need to process them once before!
    print("Train images: ", len(train_images))
    print("Valid images: ", len(valid_images))
    print("Test images: ", len(test_images))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(train_images[i])
        plt.axis("off")
        plt.title(y_train[i])
    plt.show()

    print("Time elapsed: ", time.time() - start)
    
    # Preprocessing
    IM_WIDTH, IM_HEIGHT = (64, 64)
    print("Converting images to size {} x {}...".format(IM_WIDTH, IM_HEIGHT))

    start = time.time()
    train_images = resize(train_images, IM_WIDTH, IM_HEIGHT)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(train_images[i])
        plt.axis("off")
        plt.title(y_train[i])
    plt.show()
    print("Time elapsed: ", time.time() - start)


    print("Compute Time: ", round(time.time() - start), "s")


if __name__ == '__main__':
    main()
