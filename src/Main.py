from matplotlib import pyplot as plt
from src.utils import ReadFile
import time


def main():
    start = time.time()
    test_images, train_images, valid_images = ReadFile.read_images(read_from_processed=True) # you need to process them once before!
    print("Train images: ", len(train_images))
    print("Valid images: ", len(valid_images))
    print("Test images: ", len(test_images))
    plt.imshow(train_images[6][:, :, :])
    plt.show()

    print("Compute Time: ", round(time.time() - start), "s")


if __name__ == '__main__':
    main()
