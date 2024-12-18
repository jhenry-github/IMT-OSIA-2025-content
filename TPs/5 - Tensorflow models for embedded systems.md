# Basics of Tensorflow vs. scikit learn

We have worked with scikit learn in the previous labs, because working with Tensorflow without having covered neural networks makes the process uncomfortable. However, once that you understand the structure of a neural network, getting to tensorflow from scikit learn is not very difficult.

## Linear Regression

Let's start with a simple data set, supposed to represent the prices of houses based on their square footage. We pick 5 houses, of prices from 1000 to 3000 (these are square feet, divide by 9 if you want in square meters), and we select their matching price. To create a bit of randomness, we add some random value to these prices, so the set does not form a perfect straight line.

```shell
# 1. Simple data with noise
np.random.seed(42)  # Set seed for reproducibility
X = np.array([[1000], [1500], [2000], [2500], [3000]], dtype=float)  # House sizes
y_true = np.array([200, 250, 300, 350, 400], dtype=float)  # True house prices

# Introduce noise in the y values
noise = np.random.normal(0, 15, size=y_true.shape)  # Noise with mean=0, std=15
y = y_true + noise  # Add noise to true values

```

Let's plot these houses areas and their prices so you can see what we are working from.

```shell
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

# Plot original data points with noise
plt.scatter(X, y, color='blue', label='Noisy Data Points')

# Add labels and legend
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price (in thousands)')
plt.title('Linear Regression: original data')
plt.legend()
plt.grid(True)
plt.show()
```

Nothing extraordinary so far. You may remember that we used the normal equation to solve the problem with scikit learn, something like this:

```shell
# Import libraries
from sklearn.linear_model import LinearRegression

# 2. Initialize the linear regression model
model = LinearRegression()

# 3. Train (fit) the model
model.fit(X, y)

# 4. Predict house prices
predicted_sklearn = model.predict(X)

# 5. Display results
print("Coefficients (Weight):", model.coef_)
print("Intercept (Bias):", model.intercept_)
print("Predicted Prices:", predicted_sklearn)
```

There is no gradient descent, because the normal equation solves the problem in one go. However, it is computationally expensive, so you would only run this type of structure on a small problem. Although you can find the version of tensorflow with the normal equation in the class material, in practice we use gradient descent because the process allows us to deal with much larger datasets (as we can load part of the dataset as we train, while the normal equation requries the full dataset to be injected in memory).

So let's generate a larger dataset, of 10,000 houses, and plot their prices.


```shell
import matplotlib.pyplot as plt

# Generate data (large dataset example)
np.random.seed(42)
X = np.random.rand(10000, 1) * 1000  # 10,000 house sizes (random values)
y = 200 + 0.1 * X + np.random.randn(10000, 1) * 20  # True linear function with noise

# Plot the data 
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.2, label='Noisy Data Points')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price (in thousands)')
plt.title('Large dataset of house prices')
plt.legend()
plt.grid(True)
plt.show()
```

Now let's use Tensorflow. If you want to, you can copy the code above and try scikit learn with the normal equation. It should still work, but you should see that the machine starts to slow down a bit.


Meanwhile, with tensorflow. we first load the libraries, then we use Standard Scaler to scale x a,d y to the same units. In the small example above, you can see that (in units) the size of the houses is 10 times bigger than the price. This will cause distorsion (bias). So the standard scaler makes sure that all parameters are on the same scale. The tool retains the scaling ratio, so it is easy to convert back when needed. This is especially important when you have many parameters, because you are almost guaranteed that some parameters will have large scales that will make other parameters (with smaller scales) inconsequent.

```shell
# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Scale the data for TensorFlow training
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)  # Scale features
y_scaled = scaler_y.fit_transform(y_true)  # Scale targets
```

In tensorflow, (almost) everything is a neural network. Tensorflow uses keras as a wrapper to define the model with simple words. So here, our model is a linear regression model, but it is also a neural network. Because there is a single parameter (the house area) that we inject into the model, the input structure is a 1D vector (of one value). We also want to initialize the parameters to 0. Now, as everythgin is a neural network, we define our model as a 'sequential' model, i.e. a model where each layer is computed then sent to the next one. This is just a polite way of following tensorflow way of live, because our network has a single layer (the Dense layer) with a single neuron where the equation will reside.


```shell
#  Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), kernel_initializer='zeros', bias_initializer='zeros')
])
```
The dense layer is fundamental in neural networks, it is often there, and it connects the input (in our case, the house area) to the output (in our case, the predicted price). The job of he dense layer is to perfrome the activation function (in our case, the linear equation), with output = activation(input . weights + some value - maybe 0).

Once we have defined the model, we need to compile it. Compiling may have a different meaning here than in programming. In our context, it means binding the optimizer (ink the optimization algorithm (e.g., adam, sgd) to the model), specifying the loss function (define how the error between predictions and true labels is calculated, and setting metrics (allow the model to report useful evaluation measures (like accuracy)). So compiling is just about initializing the chosen optimizer, preparing the loss function to use during training and configuring any additional metrics to track performances.


In our case, it is a linear regression case, so we use Stochastic Gradient Descent as the optimizer. Let's choose an initial learning rate of 0.01. THis will probably need to be refined later. We also use Mean Square Error (mse) as our metric to compare the calculated values to the real price values.


```shell
#  Compile the model with SGD optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
```

Once the compilation is done, we need to train the model, which is the main task. Let' use 100 epochs and a bacth size of 32 houses at each pass. I use verbose=0 so that the system does not tell you the result of each pass, but set it to 1 if you want to see what happens (warning: verbose means that it is going to tell you each operation during the training phase).

Once the training completes, let's save the mdeol, it is always a good habit to take. The model is in memory, but the scond you close the notebook, you lose the training you spent so much time running. So saving a model you will decide to discard later is always betetr than not saving and having to reinvest, possibly hundreds of hours, in training.

```shell
# Train and save the model
history = model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, verbose=0)
model.save("tf_simple_linear_regression_model.keras")
print("Model saved successfully!")

```

The last 2 operations consist first in predicting all the values in the training set (now that the model is trained) and saving them somewhere, then converting them back to their original scale. This way we can look at the data and check the efficiency of the model. The other opration is to extract from the model the slope of the curve (or weight) and the intercept (or bias), so we can look at the line equation.


```shell
#  Predict values
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Reverse scaling to original scale

#  Extract weights and bias
weights, bias = model.layers[0].get_weights()
print("Weight (Coefficient):", weights[0][0])
print("Bias (Intercept):", bias[0] * scaler_y.scale_[0] + scaler_y.mean_)  # Adjust bias for scaling

```

Let's plot the result. You may not be super impressed. With our (random) learning rate and epoch count, we are not expecting any miracle.

```shell
# Plot the results
plt.figure(figsize=(10, 6))

# Scatter plot for noisy data
plt.scatter(X, y_true, color='blue', alpha=0.2, label='Noisy Data Points')

# Plot the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Best Model Regression Line')

# Add labels, legend, and title
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price (in thousands)')
plt.title('Linear Regression with un-refined tf')
plt.legend()
plt.grid(True)
plt.show()
```

In real life, we want to find the right learning rate and ecpoh number for any problem we are training on. One good tool is keras tuner. In tuner, you define a model as we did above, but it is called a super model, because in addition to the model structure, you add a range for the learning rate, and you log each learning rate and its results.

```shell
# Define the HyperModel for Keras Tuner
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,), kernel_initializer='zeros', bias_initializer='zeros')
    ])
    # Tune the learning rate
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-1, sampling="log")
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse')
    return model
```

You also want to include another block of code to try multiple epoch values, for example from 50 to 200, by change of 10 epochs. Then you tell the tuner to test them and store the results.

```shell
#  Custom Tuner to include epoch tuning
class RegressionTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Add epochs as a tunable hyperparameter
        hp = trial.hyperparameters
        epochs = hp.Int("epochs", min_value=50, max_value=200, step=10)
        
        # Run the training with the selected epoch count
        kwargs['epochs'] = epochs
        results = super(RegressionTuner, self).run_trial(trial, *args, **kwargs)
        
        # Return the final metric for the trial
        return results
```

With these two functions defined, you can instantiate the tuner, with the objective of finding the best learning rate and epoch count (that minimizes the loss function), and try 10 learning rates and epochs combinations.

```shell
#  Instantiate the tuner
tuner = RegressionTuner(
    build_model,
    objective="loss",  # Minimize loss
    max_trials=10,     # Number of trials
    executions_per_trial=1,
    directory="sgd_tuning_epochs",
    project_name="regression_sgd_epochs"
)
```
You can also ask the tuner to stop if it finds a super value. The you run the search, which takes some time. At the end of the search, you pick the best learning rate and epoch combination.


```shell
# Define a callback for early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

# Run the hyperparameter search
tuner.search(X_scaled, y_scaled, batch_size=32, callbacks=[stop_early], verbose=1)

#  Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best Learning Rate:", best_hp.get("learning_rate"))
print("Best Epoch Count:", best_hp.get("epochs"))
```

Once you have done that, you can use these parameters found with keras tuner to train your model, and you should get much better results.

In real life, all these steps are done in one go, for example like this (I include the initial random data generation part, in real life this would be where you load your dataset):


```shell
# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt

# 1. Generate data (large dataset example)
np.random.seed(42)
X = np.random.rand(10000, 1) * 1000  # 10,000 house sizes (random values)
y_true = 200 + 0.1 * X + np.random.randn(10000, 1) * 20  # True linear function with noise

# 2. Scale the data for TensorFlow training
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)  # Scale features
y_scaled = scaler_y.fit_transform(y_true)  # Scale targets

# 3. Define the HyperModel for Keras Tuner
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,), kernel_initializer='zeros', bias_initializer='zeros')
    ])
    # Tune the learning rate
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-1, sampling="log")
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

# 4. Custom Tuner to include epoch tuning
class RegressionTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Add epochs as a tunable hyperparameter
        hp = trial.hyperparameters
        epochs = hp.Int("epochs", min_value=50, max_value=200, step=10)
        
        # Run the training with the selected epoch count
        kwargs['epochs'] = epochs
        results = super(RegressionTuner, self).run_trial(trial, *args, **kwargs)
        
        # Return the final metric for the trial
        return results

# 5. Instantiate the tuner
tuner = RegressionTuner(
    build_model,
    objective="loss",  # Minimize loss
    max_trials=10,     # Number of trials
    executions_per_trial=1,
    directory="sgd_tuning_epochs",
    project_name="regression_sgd_epochs"
)

# 6. Define a callback for early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

# 7. Run the hyperparameter search
tuner.search(X_scaled, y_scaled, batch_size=32, callbacks=[stop_early], verbose=1)

# 8. Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best Learning Rate:", best_hp.get("learning_rate"))
print("Best Epoch Count:", best_hp.get("epochs"))

# 9. Train the model with the best hyperparameters
best_model = tuner.get_best_models(1)[0]
best_model.fit(X_scaled, y_scaled, epochs=best_hp.get("epochs"), batch_size=32, verbose=0)

# 10. Make predictions
y_pred_scaled = best_model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 11. Extract weights and bias
weights, bias = best_model.layers[0].get_weights()
print("Weight (Coefficient):", weights[0][0])
print("Bias (Intercept):", bias[0] * scaler_y.scale_[0] + scaler_y.mean_)

# 12. Plot the results
plt.figure(figsize=(10, 6))

# Scatter plot for noisy data
plt.scatter(X, y_true, color='blue', alpha=0.2, label='Noisy Data Points')

# Plot the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Best Model Regression Line')

# Add labels, legend, and title
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price (in thousands)')
plt.title('Linear Regression Using SGD with Tuned Learning Rate and Epochs')
plt.legend()
plt.grid(True)
plt.show()

```

## Logistic Regression

Now that you get a grasp of tensorflow structure, we can go a bit faster. You may remember the idea of simple logistic regression, i.e. finding membership between two groups with a probbaility outcome, through the sigmoid function. Here again, let's generate some random points so we have something to work from. We first import the libraries we need, then we generate 1000 points, that we immediately randomly assign to one group or another (we do that so we can later evaluate the peformance of our algorithm; in real life, of course, you would likely not know the membership). We also use the usual random state fixed value, so you get the same random distribution as me when you run this exercize. 

```shell
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1. Generate synthetic binary classification data
X, y = make_classification(n_samples=1000,    # Number of samples
                           n_features=2,     # Number of features (2 for easy visualization)
                           n_classes=2,      # Binary classification
                           n_informative=2,  # Number of informative features
                           n_redundant=0,    # No redundant features
                           random_state=42)  # Reproducibility

# 2. Plot the data
plt.figure(figsize=(8, 6))

# Scatter plot of the two classes
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', alpha=0.6, edgecolor='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.6, edgecolor='k')

# Add labels, legend, and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Binary Classification Data')
plt.legend()
plt.grid(True)
plt.show()
```


With scikit learn, we saw that the procedure was quite straightforward. After loading the libraries, we need data. You can reuse the data from above, so step 1 is not necessary (but it is here to remind you that you need to load the data). Then, as usual, we split the data into training and test sets. 

Then we just call the logistic regression model, and ask the algorithm to find the best values (fit), given our training set of positions. We then ask the model to compute its prediction on the test set. As we know which point belongs to which group, we can also ask the model to print the prediction accuracy.





```shell
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Generate synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                           n_informative=2, n_redundant=0, random_state=42)

# 2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

In my attempt, the prediction accuracy is at 88%, not bad. Let's graph the points and the decision boundary, along with the points in each group, to see where the model is right, and where it has some difficulties.

```shell
# 6. Plot decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.6, edgecolor='k')
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary (Scikit-learn)')
plt.show()
```

It should not be suprising to see that the model has a hard time with the outliers we randomly placed in the wrong group when generating the random points.


Let's look at the same procedure with tensorflow. The initial part is exactly the same as with scikit learn, we load libraries (okay, we need tf libraries instead of sk-learn libraries), then we get some data, and we split the data into training and test sets.


```shell
# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Generate synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                           n_informative=2, n_redundant=0, random_state=42)

# 2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Let's then build the model. (almost) everything in tensorflow is a neural network, so we build a neural network of type sequential, i.e. each layer will be processed in order, and the output of one layer will be sent to the next layer. Practically, this is just a cosmetic wording, because in the logistic regression model we have a single equation, so our neural network is one layer structure, with one neuron. So 'sequential' is the default, but it does not bring any constraints, as there is a single cell anyway. The input is two-dimensional (because our points have (x,y) coordinates), so input has 2 features. Then the input is passed onto the main, Dense layer, wehre there is a single neuron, that uses the sigmoid function.



```shell
# 3. Build the TensorFlow model with Input layer
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),  # Explicit Input layer for 2 features
    tf.keras.layers.Dense(1, activation='sigmoid')  # Logistic Regression layer
])
```


The next step is to compile the model, i.e. tie together the model, the optimization function, the loss function and a mechanism to evaluate the accuracy (the loss) as we train. The optimization function is stochastic gradient descent. The loss function is the binary_crossentropy function that computes the distance between the predicted value and the real value (you may remember from the class notes that this uses an equation in the form loss = -1(1/n). sum(y.log(p)+(1-y).log(1-p)). Then the metric is the accuracy, i.e. whether the prediction (group 0 or 1) matches the real value.

```shell
# 4. Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
```

Once the model is compiled, we can go ahead and train it. Just like in the previosu case, we use, as a starting point, 100 epochs, and a batch size of 32 points per run. If you want to see the details of the training steps, set the verbose option to 1.

As usual also, we save the model as soon as we have it. Again, with large models and large datasets, it is easier to save a model, then delete the file if we don't need it, than forget to save it, think the model is 'there' because it is in memory, then realise after closing and re-opening Jupyter that the model was not saved and we have to re-train.


```shell
# 5. Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
model.save("logistic_regression_model.keras")
print("Model saved successfully!")
```

Our next step is to evaluate the model. This is done by running the model on each point of the test set. We then print the accuracy value.

Because it is a categorical prediciton (group 0 or 1), it is always useful to display a confusion matrix, to see if we can find a pattern for the points that are misclassified. In this particular case, because we use random data, we have about the same number of outliers (points in the wrong group) for both groups, but in real life, you may see one group that is performing worse than the other, which may lead you to look at your data more closely to understand why, and whether or not you need to work on your data a bit more.


```shell
# 6. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", accuracy)

# 7. Make predictions
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# 8. Print evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
Once we have our model and its accuracy, we can visualize the points, their group membership and the location where the model placed the decision boundary, just like we did fro scikit learn.

```shell
# 9. Plot decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.6, edgecolor='k')

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (Z > 0.5).astype(int)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary (TensorFlow)')
plt.show()
```

## Saving to tflite

The reason for us to look at tensorflow is because most IoT boards and embedded systems cannot run full Python and cannot run complex structures like scikit learn. Therefore, once you have a model trained 'in the cloud', deploying it on the IoT edge is a major problem. However, tensorflow has developed a light version, tflite, that can run on the edge. The process is to first develop a full model as we did above. You then save the model into a tflite format, using a TFLiteConverter function. The result is a .tflite file containing the model optimized for inference. Once this phase completes, the .tflite model file is copied to the IoT or edge device.
TFLite models are designed to run efficiently on lightweight devices like microcontrollers, edge CPUs, GPUs, or specialized NPUs, because they only contain the parameters needed for the inference (i.e., the equations, but not the parameters and all the other meta structure of the model - this means that you can't really use the tflite model for further training, just for inference).

On the IoT or edge device, a TFLite interpreter binary runs the .tflite model. This interpreter is typically written in C++ (or C) and is highly optimized for low-memory and low-power environments.
When developing an application for the edge device, you write code in C++, C, or another supported language to call the TFLite interpreter and run inference on the model. The interpreter loads the .tflite file, processes input data, and generates predictions.

Therefore it is useful to be able to convert the tensorflow model into a tflite model.

First, let's look at the logistic regression model. We use 'os' to check the model file size, and summary to look at the tensorflow parameters for this model.


```shell
import os
# 1. Check the file size
model_path = "logistic_regression_model.keras"
file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
print(f"Model file size: {file_size:.2f} MB")

# 2. Load the model
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# 3. Examine the model
loaded_model.summary()
```

As you can see, the model is fairly small, it would run easily on a constrained board, if the board could run some complex OS, with Python support etc. However, as the firmware is likely compact, it will likley call the tflite interperter, so we need the tflite model. Let's save the model into tflite format.


The model may be in memory, in which case you can start from step 2. But if you saved the model and closed Jupyter, use step 1 to reload the model into memory


```shell
import tensorflow as tf

# Path to the saved Keras model
model_path = "logistic_regression_model.keras"

# 1. Load the Keras model
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# 2. Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)  # Create the converter
tflite_model = converter.convert()  # Convert the model to TFLite format

# 3. Save the TFLite model to a file
tflite_model_path = "logistic_regression_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {tflite_model_path}")

```

The file itself is fairly small, but the initial tensorflow model was small as well. Let's look at the parameters that the tflite file contains.



```shell
interpreter = tf.lite.Interpreter(model_path="logistic_regression_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite Model Input Details:")
print(input_details)

print("\nTFLite Model Output Details:")
print(output_details)
```

For now, note that the quantization is not set (we do not change the precision of the calculations, we'll do that for the CNN part below).

Let's also look at the linear regression (tensorflow) model. The process is the same as for the logistic structure.


```shell
import os
# 1. Check the file size
model_path = "tf_simple_linear_regression_model.keras"
file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
print(f"Model file size: {file_size:.2f} MB")

# 2. Load the model
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# 3. Examine the model
loaded_model.summary()
```

Here again, the model is small, but we need a tflite structure to run at the edge. Let's convert it.

```shell
import tensorflow as tf

# Path to the saved Keras model
model_path = "tf_simple_linear_regression_model.keras"

# 1. Load the Keras model
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# 2. Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)  # Create the converter
tflite_model = converter.convert()  # Convert the model to TFLite format

# 3. Save the TFLite model to a file
tflite_model_path = "simple_linear_regression_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {tflite_model_path}")
```

And let's look at the tflite file structure.


```shell
interpreter = tf.lite.Interpreter(model_path="simple_linear_regression_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite Model Input Details:")
print(input_details)

print("\nTFLite Model Output Details:")
print(output_details)
```

Here again the structure is light. In most cases however, the model is large, because the dataset is large and the model is complex, and there will need to be additional steps in the translation to tflite. let's explore a simplified case below, with images and a CNN.


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
