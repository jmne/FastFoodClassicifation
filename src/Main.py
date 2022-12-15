from matplotlib import pyplot as plt
from models import Model1
from utils import ReadFile
import time


def main():
    """
    Example application that processes a set of images.

    Args:
    """
    start = time.time()
    print("Starting reading images")
    test_images, train_images, valid_images = ReadFile.read_images(
        read_from_processed=False)  # you need to process them once before!
    print("Train images: ", len(train_images))
    print("Valid images: ", len(valid_images))
    print("Test images: ", len(test_images))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(train_images[i])
        plt.axis("off")
    plt.show()

    print("Time elapsed: ", time.time() - start)
    print("Starting model...")

    Model1.model()

    print("Compute Time: ", round(time.time() - start), "s")


if __name__ == '__main__':
    main()
