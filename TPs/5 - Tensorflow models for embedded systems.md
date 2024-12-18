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

The result is then sent to another convolutional layer with 64 filters of size 3x3. That layer detects more complex patterns by building on the features detected by the first convolutional layer.

After that second convolutional layer, the result is pooled again, reducing the feature maps again (e.g., 16x16 → 8x8).

The outcome is sent to a flatenning layer. As its name indicates, that layer flattens the 2D feature maps into a 1D vector so that it can be fed into dense (fully connected) layers.

The next layer is therefore a dense with 64 neurons and ReLU activation. The goal of that layer is to learn high-level patterns in the data after flattening.

The output layer is then a classifier with 10 neurons, one for each of the CIFAR-10 classes. The 'softmax' function converts the output of each neuron into a probability.
The system outputs 10 probabilities (one per neuron), the one with the highest probability is declared the predicted class. The code looks like this:



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

In short, this is the structure of your CNN:


```shell
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input (InputLayer)          [(None, 32, 32, 3)]       0         
 conv2d (Conv2D)             (None, 30, 30, 32)        896       
 max_pooling2d (MaxPooling2D) (None, 15, 15, 32)       0         
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
 max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)         0         
 flatten (Flatten)           (None, 2304)              0         
 dense (Dense)               (None, 64)                147520    
 dense_1 (Dense)             (None, 10)                650       
=================================================================
Total params: 167,562
Trainable params: 167,562
Non-trainable params: 0
_________________________________________________________________
```

Once the model is defined, the next step is to compile it. The compiler uses 'adam', an adaptive optimizer that adjusts learning rates automatically during training. Remember that nale, adam is very efficient and widely used for CNNs.

The loss function is crossentropy, because the system outputs probabilities (sigmoid functions), and categorical because it is a multi-class classification. During the training phase, this compilation will be used to masure the difference between the predicted probabilities and the true labels, with the metric 'accuracy', i.e. the percentage of correctly classified samples. 

```shell
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

You can then go on to train the model on the training set, with verification against the validation set. You could just use 'model.fit', but storing the result in 'history' allows you to reuse the result in other blocks. The training uses in this example 10 epochs, but you can try a different number if you want to see the difference. Depending on your machine, the trianing may take some time. 

```shell
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

Once the model is trained, a natural next step is to evaluate it. While we are there, let's also visualize the training and validation accuracy over the training epochs. You should see that the accuracy increases overall, both for the training and the test sets. 


```shell
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Plotting training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
```

If you kept the default 10 epochs, the final acuracy value, displayed just below the Jupyter block (and above the graph), may not be great. You could improve it by incrfeasing the number of epoch, and with keras.tuner by searching for the best parameters. But our goal is to underatand the structure, not really to get the best possible model.

Another option, when working on well-studied problems (like image recognition on common images, like here, but also on other elements like well-known command words in the English language, like 'up' vs. 'down' etc.) is to look for a pre-trained model. The advantage is that the pre-trained model may have been trained on a dataset similar to yours, but much larger. You may still need to train that model on your data, but the training is akin to ftransfer learning, where you apply the knowledge of your model on a new, but not so different, dataset, so the model just need to learn 'what is new' instead of 'everything'. The process is much faster. VGG16 has already learned rich feature representations (e.g., edges, shapes, textures), and using it as a feature extractor saves time and computational resources compared to training a new CNN from scratch.

In the case of image recognition, a well-known model is the VGG16 model, a pre-trained Convolutional Neural Network trained on the ImageNet dataset (1.2 million images, 1000 classes). Training that model took a while and big machines, so using it will save your machine time. 

So a first step is to load that new model (fortunately, you can load it directly from tensorflow). VGG16 was originally trained on 224x224 images, but because our dataset has 32x32x3 images, we are asking tensorflow to resize internally to match our 32x32x3 structure. With include_top=false, we exclude the fully connected layers at the top of the VGG16 model, leaving only the convolutional base, which extracts features from images (which is the part we care about). We also load weights pre-trained on the ImageNet dataset.

```shell
base_model = tf.keras.applications.VGG16(input_shape=(32, 32, 3),
                                         include_top=False,
                                         weights='imagenet')
base_model.trainable = False  # Freeze the base model
```

After loading that model, we freeze it. Freezing prevents the weights of the pre-trained VGG16 convolutional layers from being updated during training. So we do not change the pre-learned features from the model. The convolutional base will then act as a fixed feature extractor. This is a very common way of reusing a large model.


On top of the VGG16 model, we add a few layers. One layer flattens the feature maps (the output from VGG16) into a 1D vector for input to the dense layers. Then, we add a a fully connected layer with 64 neurons and ReLU activation to learn new features. Then, the result gets as before into a 10 neuron classifier with softmz to give us the probability for each of our 10 classes. In short, VGG16 does most of the heavy work, and we just convert its output into a 1D vector, then to a standard dense + class structure to get the probability.

```shell
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Once this new model is defined, we can compile it. The structure here is similar as the one we had above.

```shell
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

Training will take longer than the previous tiny mode we worked with. On my machine, it is about 10 times slower than the small model above, but the model is much larger, so this is a good tradeoff. 

Once the training completes, we can plot again the training and validation accuracy. As the code is the same as above, let's add something new, the training vs validaiton loss, another very common graph people use. This graph shows the gap between training and validation loss indicates potential overfitting.

```shell
# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```

You may be very happy with this new code, or slightly dispointed if its performances are only marginally different (or worse) than the lighter model. Keep in mind that we are focusing in this lab on the structures. In real life, you would have used keras tuner or another optimizer to find the best parameters for your light model, and your accuracy would have reached a plateau (for example 75%) without hope of going further. You would not have wanted to train a large model from scratch, so you would have done what we did above with VGG16, with again an optimizer to get to the best parameters. This time, you accuracy would have reached something acceptable, like (for example) 92%. These percentages are typical of scores people get in contests with CIFAR-10. If you have time, try to use keras tuner to see if you can boost the performances of both models. 

The next step is of course to test the model. One fun way to do it is to pick a random image from the test set, and check what the model finds. Let's first load the test set, and assign words to labels (remember, with one-hot encoding, we have category numbers, but names are better).

```shell
print('hello world')# Class names for CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()  # Only use test data

# Normalize the test data
x_test = x_test / 255.0
```

Next, we pick a random image from the test set, format it to the way the model expects, apply the model prediction to it, then display the image and the prediction.

```shell
# Select a random image from the test set
random_index = random.randint(0, len(x_test) - 1)  # Random index
random_image = x_test[random_index]
true_label = y_test[random_index][0]  # True class label (integer)

# Expand dimensions to match the model's input shape
image_batch = np.expand_dims(random_image, axis=0)  # Shape: (1, 32, 32, 3)

# Make predictions
predictions = model.predict(image_batch)
predicted_class = np.argmax(predictions[0])  # Index of the highest probability
predicted_label = class_names[predicted_class]
true_label_name = class_names[true_label]

# Display the image with the prediction and true label
plt.figure(figsize=(2, 2))  # Adjust figure size here (smaller size)
plt.imshow(random_image)
plt.title(f"True: {true_label_name}\nPredicted: {predicted_label} ({predictions[0][predicted_class]:.2f})")
plt.axis('off')
plt.show()

```

You can run this code a few times, each time a random image is picked. You should see that the model accuracy is often good, but not always.

It is often interesting, when we do not have too many categories, to display the confusion matrix, to check which image is always right, and which one is often wrong. And when its wrong, with what other label is the image usually confused.

```shell
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for CIFAR-10')
plt.show()
```
 In my training, the worst confusion comes from cats, dogs, frogs and birds, that are often confused with one another. In real life, more training, and more images of these offenders would help increase the accuracy.


 Now that you understand the model, let's save it. There are multiple possible formats for AIML models. From tensorflow, the most common format is kears:

```shell
# Save the model in the Keras format
model.save("cifar10_cnn_model.keras")
print("Model saved successfully in Keras format!")

```
 
Keras is supposed to be more than Tensorflow, but not every structure understand it. A more common structure is HDF5:

```shell
# Save the model in HDF5 format
model.save("cifar10_cnn_model.h5")
print("Model saved successfully in HDF5 format!")

```

Tensorflow dows not like it much when you use HDF5, because it prefers Keras, but HDF5 is reusable in many other, competing, structures, like Pytorch.

However, as we work with embedded systems, another good choice is to save the model in SavedModel format:

```shell
model.export("cifar10_saved_model")
print("Model successfully saved in SavedModel format!")
```

This creates a directory cifar10_saved_model/ with saved_model.pb (the model's architecture and metadata) and variables (the model's weights).

You can then move these models to other machines and load any of them with the command loaded_model = tf.keras.models.load_model("model name"), and ither .keras, .h5 extension, or just the main directory name for the SavedModel format.


If you want to run these models on an IoT board like the one we use in class, the full TensorFlow model is likely too heavy (for the memory, but also for the code you run on the board). You probably want to convert the model to tflite. You need to use the tensorflow converter, and to quantize the output (to reduce its size further). There is some loss in that operation, but the goal is not to continue training the model (in fact, you can't really continue training well from a tflite model), but just run inferences. So the arbitration will be 'how much do you accept to quantize down' vs. 'how much space do you have on your board'.


```shell
# Convert the model to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable post-training dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quantized_model = converter.convert()

# Save the quantized TFLite model
tflite_model_path = "cifar10_cnn_model_quantized.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_quantized_model)
print("Quantized TFLite model successfully saved!")
```



Let's look at the original tensorflow model, to compare it to the tflite version. Let's first load the models. Then we use for each of them the function 'summary' to get the main parameters. We also use the os library to read the size of the model in your machine.

```shell
keras_model_path = "cifar10_cnn_model.keras"
tflite_model_path = "cifar10_cnn_model_quantized.tflite"

# --- Function to Display Model Information ---
def display_model_info(model_path, format_type):
    if format_type == "Keras":
        # Load the Keras model and display summary
        keras_model = load_model(model_path)
        keras_model.summary()
        
        # Model size in MB
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        
        # Number of parameters
        num_params = keras_model.count_params()
        return {"Size (MB)": model_size, "Parameters": num_params, "Format": format_type}
    
    elif format_type == "TFLite":
        # Get TFLite model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        
        # TFLite model does not explicitly show parameters like Keras, but we report its size
        return {"Size (MB)": model_size, "Parameters": "N/A", "Format": format_type}

# --- Compare Keras and TFLite Models ---
keras_info = display_model_info(keras_model_path, "Keras")
tflite_info = display_model_info(tflite_model_path, "TFLite")

# Display results side by side
import pandas as pd

comparison_df = pd.DataFrame([keras_info, tflite_info])
print("\nModel Comparison:")
print(comparison_df)
```


You should see that the Tensorflow model is much larger than the tflite version. In fact, by playing with the quantization command, you can vary the size of the tflite output by quite a lot. You should also see, for the VGG16-derived model, the parameters saved in the file for each of the layer types. The tflite model does not have any of these parameters. If just contains the equations necessary to run the neural network, without a separate structure for the parameters (as the goal is not to continue training, just run inferences).

If you want to look further at your tflite model, you may want to print what inputs it is taking, what output it is generating, the weights it is using,e tc. in short all the information saved in the model (outside of the equations themselves) to be able to run.


```shell
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="cifar10_cnn_model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite Model Input Details:")
print(input_details)

print("\nTFLite Model Output Details:")
print(output_details)
```

You should see that the information saved is minimal, making the model very light.
