from matplotlib import pyplot as plt
from models import Model1, Model2
from utils import ReadFile
import time
from random import randrange


def main():
    """
    Example application that processes a set of images.

    Args:
    """
    start = time.time()
    print("Start reading images")
    test_images, train_images, valid_images, y_test, y_train, y_valid = ReadFile.read_images(
        read_from_processed=True)  # you need to process them once before!
    print("Train images: ", len(train_images))
    print("Valid images: ", len(valid_images))
    print("Test images: ", len(test_images))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        rand = randrange(len(train_images)-1)
        plt.imshow(train_images[rand])
        plt.axis("off")
        plt.title(y_train[rand])
    plt.show()

    print("Time elapsed: ", round(time.time() - start), "s")
    print("Starting model...")

    # Model1.model()
    Model2.model()

    print("Compute Time: ", round(time.time() - start), "s")


if __name__ == '__main__':
    main()
