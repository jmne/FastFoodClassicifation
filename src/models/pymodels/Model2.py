from utils import ReadFile

# Common
import time
from matplotlib import pyplot as plt
import numpy as np
import math
from PIL import Image
import pandas as pd
import seaborn as sns

# Model
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

# Given a numpy array of images, resizes all of the images to the specified width and height and returns the standardized images
def resize(images, width, height):
    """
    Resizes images to the specified width and height.

    Args:
        images: write your description
        width: write your description
        height: write your description
    """
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

# Resizes the images and normalizes them
def preprocess(images, width, height):
    """
    Resize and preprocess image.

    Args:
        images: write your description
        width: write your description
        height: write your description
    """
    # Preprocessing: Resize
    print("Converting test images to size {} x {}...".format(width, height))
    start = time.time()
    images = resize(images, width, height)
    print("Time elapsed: ", time.time() - start)
    # Preprocessing: Normalize
    images = images / 255.0
    return images

# The model
def model():
    """
    Train the lenet5 model.

    Args:
    """
    # Specify size of image data to be inputted into model
    IM_WIDTH, IM_HEIGHT = (100, 100)

    # Read in and load the training and testing images, along with their labels
    start = time.time()
    print("Starting reading images")
    test_images, train_images, valid_images, test_labels, train_labels, valid_labels = ReadFile.read_images(
        read_from_processed=True)  # you need to process them once before!
    print("Train images: ", len(train_images))
    print("Valid images: ", len(valid_images))
    print("Test images: ", len(test_images))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(train_images[i])
        plt.axis("off")
        plt.title(train_labels[i])
    plt.show()

    print("Time elapsed: ", time.time() - start)

    # Create a list of all the unique labels
    labels = list(set(train_labels))
        
    # Build np arrays for the y data (integer values representing labels)
    y_train = np.asarray([labels.index(x) for x in train_labels])
    y_test = np.asarray([labels.index(x) for x in test_labels])
    y_valid = np.asarray([labels.index(x) for x in valid_labels])

    # Preprocess training and testing data
    x_train_tf = preprocess(train_images, IM_WIDTH, IM_HEIGHT)
    x_test_tf = preprocess(test_images, IM_WIDTH, IM_HEIGHT)

    # Creating the model
    lenet5 = keras.Sequential([
        keras.layers.Conv2D(input_shape=(IM_WIDTH, IM_HEIGHT, 3), filters=6, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
        keras.layers.AveragePooling2D(pool_size=2, strides=2),
        keras.layers.Conv2D(16, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
        keras.layers.AveragePooling2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=tf.nn.relu),
        keras.layers.Dense(84, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    lenet5.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['acc'])
    lenet5.summary()

    # Train the model
    print("Starting training...")
    start = time.time()
    y_train_tf = keras.utils.to_categorical(y_train) # Convert labels from integers to binary np arrays

    print(x_train_tf.shape)
    print(y_train_tf.shape)

    lenet5.fit(x_train_tf, y_train_tf, epochs=20)
    lenet5.save('./jupyter/lenet5.h5')
    print("Completed in {} seconds".format(time.time() - start))

    # Predict images using the model and output confusion matrix
    predictions = lenet5.predict(x_test_tf)
    preds = [np.argmax(x) for x in predictions]
    correct = 0
    for i, pred in enumerate(preds):
        if pred == y_test[i]:
            correct += 1
    print('Test Accuracy of the model on the {} test images: {}% with TensorFlow'.format(x_test_tf.shape[0],100 * correct/x_test_tf.shape[0]))
    
    matrix = confusion_matrix(y_test, preds)
    sns.heatmap(matrix, annot=True)
    plt.title('Food Confusion Matrix')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()
    for i in range(len(labels)):
        print("{}: {}".format(i, labels[i]))