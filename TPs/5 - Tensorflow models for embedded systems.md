a
a
a
# CNN Example, the CIFAR-10 case
The CIFAR-10 dataset, which is a popular dataset for image classification tasks. It is a collection of 60,000 images divided into 50,000 training images and 10,000 test images
Each image is 32x32 pixels with 3 color channels (RGB). There are 10 classes representing different object categories: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.
This dataset is popular in IoT, because any low definition camera, directly on the sensor or just coupled to it, can be used to recognize objects. It is also a change from the audio files we saw in the previous lab, but with the asme level of complexity.
The goal of this part is therefore to load the dataset, and train a CNN on that dataset. In the inference phase, the goal will be to look at an image, and find which category it belogs to.
Let's start with loading the model:

```shell
# Import required libraries
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

```
This part should be easy to read at this point. Note that tensorflow.keras.datasets includes a few of pre-loaded libraries for you to play with, including cifar10. You can find more [here](https://keras.io/api/datasets/). 
The tensorflow library that we want to use is to_categorical, meaning that the labels will be converted to a one-hot encoding vector. One-hot encoding is a term you will find very often. 
In short, the process counts the categories, and creates a vector of as many bits as needed to represent all categories. For example, if you have 4 categories, the to_categorical function will create a vector of length 4.
Then, the categories will be encoded as \[1,0,0,0], \[0,1,0,0], \[0,0,1,0] and \[0,0,0,1].


Our next step is to define the CNN model. Here is a structure that works well. Without this structure, you would probbaly look around for examples of CNNs that were used succesfully on data similar to yours, then experiement from there.
keras tuner may also help you find the best structure. Here, we will use 7 layers. We first load the model, with a 'sequential' (linear) stack of layers. Each layer processes input and passes the output to the next layer.

Then, the input layer expects images in 32x32 pixels, in RGB (so, 3 colors).

The input layer is then passed into a first convolutional layer, A 2D convolution layer that applies 32 filters (feature detectors) to the input image. Each filter is a 3x3 kernel that slides over the input image to detect features (e.g., edges, patterns).
With the keyword activation='relu', the ReLU (Rectified Linear Unit) functin cleans up the data, by replacing negative values with zero.

The result is then sent to a pooling layer, that applies max pooling (max value of eac 2x2 zone is retained). This phase reduces spatial dimensions by a factor of 2 (e.g., 32x32 → 16x16).



```shell
# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),  # Explicit Input layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer
    tf.keras.layers.MaxPooling2D(2, 2),                     # First pooling layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    tf.keras.layers.MaxPooling2D(2, 2),                     # Second pooling layer
    tf.keras.layers.Flatten(),                              # Flatten the feature maps
    tf.keras.layers.Dense(64, activation='relu'),           # Fully connected layer
    tf.keras.layers.Dense(10, activation='softmax')         # Output layer
])
```







```shell
print('hello world')
```

