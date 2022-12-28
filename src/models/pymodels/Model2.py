# Importing modules
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import cv2
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential

from sklearn.model_selection import train_test_split

np.random.seed(1)


def model():
    """
    This function creates a model from the training data.

    Args:
    """
    # Processing training data
    # -> appending images in a list 'train_images'
    # -> appending labels in a list 'train_labels'

    train_images = []
    train_labels = []
    shape = (200, 200)
    train_path = '../resources/train'

    print("Processing training data..")

    for folder in os.listdir(train_path):
        for filename in os.listdir(train_path + '/' + folder):
            if filename.split('.')[1] == 'jpeg':
                img = cv2.imread(train_path + "/" + folder + "/" + filename)

                print("File: " + filename + "                ", end='\r')

                # Splitting file names and storing the labels for image in list
                train_labels.append(folder)

                # Resize all images to a specific shape
                img = cv2.resize(img, shape)

                train_images.append(img)

    print("\n")

    # Converting labels into One Hot encoded sparse matrix
    train_labels = pd.get_dummies(train_labels).values

    # Converting train_images to array
    train_images = np.array(train_images)

    # Splitting Training data into train and validation dataset
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)

    # Processing testing data -> appending images in a list 'test_images' -> appending labels in a list 'test_labels'
    # The test data contains labels as well also we are appending it to a list, but we aren't going to use it while
    # training.

    test_images = []
    test_labels = []
    shape = (200, 200)

    test_path = '../resources/test'

    print("Processing test data..")

    for folder in os.listdir(test_path):
        for filename in os.listdir(test_path + '/' + folder):
            if filename.split('.')[1] == 'jpeg':
                img = cv2.imread(test_path + "/" + folder + "/" + filename)

                print("File: " + filename + "                ", end='\r')

            # Splitting file names and storing the labels for image in list
            test_labels.append(folder)

            # Resize all images to a specific shape
            img = cv2.resize(img, shape)

            test_images.append(img)

    print("\n")

    # Converting test_images to array
    test_images = np.array(test_images)

    # Visualizing Training data
    print(train_labels[0])
    plt.imshow(train_images[0])

    # Visualizing Training data
    print(train_labels[-1])
    plt.imshow(train_images[-1])

    plt.plot()

    # Creating a Sequential model
    _model = Sequential()
    _model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='tanh', input_shape=(200, 200, 3,)))
    _model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
    _model.add(MaxPool2D(2, 2))
    _model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
    _model.add(MaxPool2D(2, 2))
    _model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))

    _model.add(Flatten())

    _model.add(Dense(20, activation='relu'))
    _model.add(Dense(15, activation='relu'))
    _model.add(Dense(10, activation='softmax'))

    _model.compile(
        loss='categorical_crossentropy',
        metrics=['acc'],
        optimizer='adam'
    )

    # Model Summary
    _model.summary()

    print("Training model...")

    # Training the model
    history = _model.fit(x_train, y_train, epochs=50, batch_size=40, validation_data=(x_val, y_val))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Evaluating model on validation data
    evaluate = _model.evaluate(x_val, y_val)
    print(evaluate)

    # Testing predictions and the actual label
    check_image = test_images[0:1]
    check_label = test_labels[0:1]

    predict = _model.predict(np.array(check_image))

    output = {0: 'Baked Potato', 1: 'Burger', 2: 'Crispy Chicken', 3: 'Donut', 4: 'Fries', 5: 'Hot Dog', 6: 'Pizza',
              7: 'Sandwich', 8: 'Taco', 9: 'Taquito'}

    print("Actual :- ", check_label[0])
    print("Predicted :- ", output[np.argmax(predict)])
