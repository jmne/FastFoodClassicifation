import os

from matplotlib import pyplot as plt
from models import Model1, Model2, Model3
from utils import ReadFile
import time
from random import randrange
import sys


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
