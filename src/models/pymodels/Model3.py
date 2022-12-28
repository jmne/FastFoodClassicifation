# Common
from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output as cls

# Data Loading
from keras.preprocessing.image import ImageDataGenerator as Idg

# Data Visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Model
from keras.models import Sequential
from plotly.io import write_image
from tensorflow_hub import KerasLayer as Kl
from keras.layers import Dense, InputLayer, Dropout

# Optimizer
from keras.optimizers import SGD
from tensorflow import keras

# Callbacks
from keras.callbacks import EarlyStopping as Es, ModelCheckpoint as Mc

print("\n")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def show_images(data, classes, plot_num, _model=None, grid=None, SIZE=(25, 25)):
    # Plot Configurations
    if grid is None:
        grid = [5, 6]
    n_rows, n_cols = grid
    n_images = n_cols * n_rows
    plt.figure(figsize=SIZE)

    # Iterate Through the data
    i = 1
    for images, labels in iter(data):

        # Select data Randomly
        _id = np.random.randint(len(images))
        image, label = tf.expand_dims(images[_id], axis=0), classes[int(labels[_id])]

        # Show Image
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(image[0])

        # make Prediction
        if _model is not None:
            prediction = _model.predict(image)[0]
            score = np.round(max(prediction), 2)
            pred = classes[np.argmax(prediction)]
            title = f"True : {label}\nPred : {pred}"
        else:
            title = label

        plt.title(title)
        plt.axis('off')
        cls()

        # Break Loop
        i += 1
        if i > n_images:
            break

    # Show Final Plot
    fig = plt
    fig.savefig(fname='../logs/plots/plot_' + str(plot_num) + '.png', format='png')
    plt.show()


def model():
    # Specify the root path.
    train_path = '../resources/train/'
    valid_path = '../resources/valid/'
    test_path = '../resources/test/'

    # Collect the class names.
    class_names = sorted(os.listdir(train_path))
    n_classes = len(class_names)

    # Print
    print("No. Classes : {}".format(n_classes))
    print("Classes     : {}".format(class_names))

    # Calculate the class distribution
    class_dis = [len(os.listdir(valid_path + name)) for name in class_names]

    # Virtualize class distribution
    fig1 = px.pie(names=class_names, values=class_dis, hole=0.3)
    fig1.update_layout({"title": {'text': "Validation Class Distribution", 'x': 0.48}})
    fig1.show()
    write_image(fig=fig1, file='../logs/plots/plot_1.png', format='png')

    fig2 = px.bar(x=class_names, y=class_dis, color=class_names)
    fig2.show()
    write_image(fig=fig2, file='../logs/plots/plot_2.png', format='png')

    # Initialize image data generator
    train_gen = Idg(rescale=1. / 255, rotation_range=10, horizontal_flip=True, vertical_flip=False)
    valid_gen = Idg(rescale=1. / 255)
    test_gen = Idg(rescale=1. / 255)

    # Load the datasets
    train_ds = train_gen.flow_from_directory(train_path, shuffle=True, batch_size=64, target_size=(256, 256),
                                             class_mode='binary')
    valid_ds = valid_gen.flow_from_directory(valid_path, shuffle=True, batch_size=32, target_size=(256, 256),
                                             class_mode='binary')
    test_ds = valid_gen.flow_from_directory(valid_path, shuffle=True, batch_size=32, target_size=(256, 256),
                                            class_mode='binary')

    show_images(data=train_ds, classes=class_names, plot_num=3)

    show_images(data=valid_ds, classes=class_names, plot_num=4)

    # Model URL
    url = "https://tfhub.dev/google/bit/m-r50x1/1"

    # Load Model
    bit = Kl(url)

    # Model Name
    model_name = "Fast-Food-Classification-BiT"

    # Model Architecture
    _model = Sequential([
        InputLayer(input_shape=(256, 256, 3)),
        bit,
        Dropout(0.2),
        Dense(n_classes, activation='softmax', kernel_initializer='zeros')
    ], name=model_name)

    # Model Summary
    _model.summary()

    # Learning Rate Scheduler
    lr = 5e-3

    lr_scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200, 300, 400],
                                                                     values=[lr * 0.1, lr * 0.01, lr * 0.001,
                                                                             lr * 0.0001])

    opt = SGD(learning_rate=lr_scheduler, momentum=0.9)

    # Compile
    _model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Callbacks
    cbs = [Es(patience=5, restore_best_weights=True), Mc(model_name + ".h5", save_best_only=True)]

    # Training
    history = _model.fit(train_ds, validation_data=valid_ds, epochs=50, callbacks=cbs)

    lc = pd.DataFrame(history.history)
    print(lc)

    lc.plot(figsize=(8, 5))
    plt.title("Learning Curve")
    plt.grid()
    fig = plt
    fig.savefig(fname='../logs/plots/plot_5.png', format='png')
    plt.show()

    show_images(data=valid_ds, classes=class_names, _model=_model, plot_num=6)

    print(_model.predict(test_ds)[0])
