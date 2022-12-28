# FastFoodClassicifation
[![](https://img.shields.io/github/issues/jmne/FastFoodClassicifation?style=flat-square)](https://github.com/jmne/FastFoodClassification/issues)
[![](https://img.shields.io/github/issues-pr/jmne/FastFoodClassicifation?style=flat-square)](https://github.com/jmne/FastFoodClassification/pulls)
[![](https://img.shields.io/github/license/jmne/FastFoodClassicifation?style=flat-square)](https://github.com/jmne/FastFoodClassification/blob/main/LICENSE)
[![](https://img.shields.io/github/languages/code-size/jmne/FastFoodClassicifation?style=flat-square)](https://github.com/jmne/FastFoodClassification/tree/main/src)<br>
[![Generate Documentation](https://github.com/jmne/FastFoodClassicifation/actions/workflows/generate-docs.yml/badge.svg)](https://github.com/jmne/FastFoodClassification/actions/workflows/generate-docs.yml)
[![Check Misspells](https://github.com/jmne/FastFoodClassicifation/actions/workflows/misspell.yml/badge.svg)](https://github.com/jmne/FastFoodClassification/actions/workflows/misspell.yml)

Image classification of Fast Food dishes with CNN and BiT-M-R50x1.

### Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Disclaimer](#disclaimer)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [References](#references)

### Introduction
Our idea is to identify fast food dishes from pictures.
The Fast Food Classification data set we use contains images of different types of fast food.

The dataset we used can be found [here](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset).

### Installation
- Clone the repository
```bash
git clone
```
- Install the requirements
```bash
pip install -r requirements.txt
```

We did run the project with **Python 3.9.7**, but feel free to try it with other versions.

### Disclaimer
The project and the models included are mainly based on the [BiT-M-R50x1 model](https://tfhub.dev/google/bit/m-r50x1/1) and Tensorflow.<br>
To speed up the training process, you should use your **GPU** (>10x faster)!
A guide to install Tensorflow with GPU support can be found [here](https://www.tensorflow.org/install/gpu).
<br>If you want to use a GPU, install the requirements_gpu.txt file instead of the requirements.txt file.
```bash
pip install -r requirements_gpu.txt
```
You **need** to install the CUDA Toolkit **v11.2** and cuDNN **v8.1** to use your GPU with Tensorflow.

### Usage
Basically just run the BiT-M-R50x1.ipynb or the CNN.ipynb notebook you cloned.

### Documentation
The project is documented within the code.

#### How does the BiT-M-R50x1 model work?
BiT-M-R50x1 is a version of the BiT (Big Transfer) model, which is a large-scale, deep learning model that has been trained on a massive dataset of images and texts from the internet. The BiT-M-R50x1 model is a variant of the BiT model that has been trained on the ImageNet and text from the internet to perform image classification and text classification tasks.

The BiT-M-R50x1 model is a convolutional neural network (CNN) that consists of multiple layers of interconnected nodes, or neurons. Each layer processes the input data and passes it on to the next layer for further processing. The layers at the beginning of the network extract low-level features from the input data, such as edges and shapes. As the data passes through the network, the layers extract increasingly complex and abstract features, such as objects and their relationships to each other.

To classify an image using the BiT-M-R50x1 model, you would feed the image into the model and it would process the image through its layers. The output of the final layer is a prediction of the class that the image belongs to, based on the features extracted by the model. The BiT-M-R50x1 model has been trained on a large dataset of images, so it has learned to recognize a wide range of objects and concepts.

#### How does the CNN model work?
A convolutional neural network (CNN) is a type of deep learning algorithm that is particularly well suited for image classification tasks. It is called a "convolutional" neural network because it uses a mathematical operation called convolution to process the input data.

The basic structure of a CNN consists of an input layer, an output layer, and one or more hidden layers. The hidden layers are made up of a series of interconnected nodes, or neurons, that perform computations on the input data.

In a CNN, the hidden layers are organized into a series of "convolutional" layers and "pooling" layers. The convolutional layers apply a series of filters to the input data, which extract different features from the data. The pooling layers reduce the dimensionality of the data by combining the output of the convolutional layers in a way that retains the most important information.

To classify an image using a CNN, you would feed the image into the input layer of the network. The image would then be processed through the convolutional and pooling layers, and the output of the final layer would be a prediction of the class that the image belongs to. The CNN has been trained on a large dataset of images, so it has learned to recognize a wide range of objects and concepts based on the features extracted by the filters in the convolutional layers.

For example, if you are training a CNN to classify images of dogs and cats, you would start by collecting a large dataset of images of dogs and cats. You would then use this dataset to train the CNN by feeding it the images and telling it which class each image belongs to. As the CNN processes the images, it will learn to recognize the features that are characteristic of dogs and cats, such as the shape of their noses or the patterns of their fur. Once the CNN is trained, you can use it to classify new images of dogs and cats by providing it with the images and having it predict the class of each image.

### Contributing
Contributions are always welcome!<br>
Feel free to open an issue or a pull request.

### License
This project is licensed under the GNU License - see the [LICENSE](https://github.com/jmne/FastFoodClassification/blob/main/LICENSE) file for details.

### Acknowledgments
- [Tensorflow](https://www.tensorflow.org/)
- [Kaggle](https://www.kaggle.com/)
- [BiT-M-R50x1](https://tfhub.dev/google/bit/m-r50x1/1)

### References

[1]: https://github.com/google-research/big_transfer <br>
[2]: https://arxiv.org/pdf/1912.11370.pdf <br>
[3]: https://www.arxiv-vanity.com/papers/1603.05027/ <br>
[4]: https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset <br
