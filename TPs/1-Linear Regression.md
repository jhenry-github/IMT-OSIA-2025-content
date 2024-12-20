# Lab setup
This lab supposes that you have Linux installed, for example in a VM, and that your Linux distribution can reach the Internet.

In may tasks in machine learning, you will be working with specific frameworks and specific librarires. It is not uncommon that a library would have specific dependencies, which are not compatible with dependencies of other libraries. For example, if you install Pytorch, it may require a specific version of numpy (a library for managing numbers and equations), while tensorflow would require a different numpy version. The outcome of this complexity is that it is common to install a virtual environment, from where you will be running a specific framework. Then, if you install a new framework, you switch to another framework.

Let's see how you would do this, as it is a very common task in aiml. In Ubuntu, venv is the tool of choice. From your Ubuntu command line enter:

```shell
sudo apt install python3-venv
```
Enter your user password when asked. Once the installation completes, you can use venv to create and manage environments. Let's create a first environment. Still in the command line, from the place where you would want to have environments created (each of them will be a folder, so your home directory could be a good location):

```shell
python3 -m venv IMTAIML
```

The operation should take a few seconds, because venv creates a lot of default elements in the environement, in addition to the environement itself. Once the operation completes, you should see a new IMTAIML folder. Change to this directory:

```shell
cd IMTAIML
```

Once there, you can activate your environment. This activation is done by calling the activate operation with the source command:

```shell
source bin/activate
```

You should see that your prompt has changed:

```shell
(IMTAIML) developer@lmvm-xfce:~/IMTAIML$
```

The parenthesis show in which environement context you are now operating. You can just run deactivate to exit this environment, and you can just call activate from another environement if you want to switch from one envirnement to another one. Let's stay with this environment. We will install a few tools, and will install other tools later from Jupyter. For now, install scikit-learn and tensorflow 

```shell
pip install scikit-learn tensorflow jupyter
```

Accept the packages proposed by the installer. A lot of dependencies need to be installed, so this will take a while.


Once the installation completes, launch Jupyter, which will be the main tool we will work from (in addition to Simplicity Studio).

```shell
jupyter-lab &
```


# Jupyter lab

When launching Jupyter, the server should launch on your machine, and the lab should open in a local browser. If the browser does not start automatically, note the URL in the command prompt, that should be in the form http://localhost:8888/lab?token= and some value. Copy this long URL into your browser to access Jupyter-lab.


![alt test](JupyterLab.png?raw=true "Jupyter Labs")

Jupyter comes in two flavors, Notebook and Lab. Notebook is the older version, and much of the new features are developped for the Lab version first, so Lab is better, especially if you want to share data (images, datasets etc.) between notebooks. However, it is okay if you decide to run Notebook instead of Lab.

At its core, Jupyter allows you to take notes in some blocks, and execute what is then seen as code in other blocks. When you are in a block, you can see at the top that the content is expected to be coded by default. Switch to markdown to see the difference.

When you have a block that is code, you can execute it in two ways: one is Shift + Enter, to directly execute the code (then move to the next block if it exists, or create a new block if the block you execute is the last one in the page). Another one is Option + Enter (on Mac, and Alt + Enter on Windows/Linux), to execute the code and insert a new block right after (which is useful if you execute a block somewhere in the middle of the page, and you want to insert a new block right after, before the next one). Note that just pressing 'Enter' moves you to the next line in the same block.

Try it! In the first block, type:

```shell
print('hello world')
```

Then type enter. You get to the next line in the same block.

Now type Shift + Enter. You should see the text below the block. Jupyter displays below the block the output of the command you are executing. Jupyter will also display warnings when applicable. Please know that you can set the level of granularity that you want Jupyter to display, but this is beyond the scope of this class (however, look it up if you are interested, starting with 'import warnings' to learn how to filter).

In the block below, type "this is a test" and press Shift + Enter.

The system should output a syntax error, as what you typed is not an interpretable Python code. Go back to that block, and set its type at the top of the page to Markdown. Press Shift Enter. This time the text should appear as text, outside of a block. This type of functionality is very useful to anotate around the commands. My suggestion: use it extensively! In 6 months, you will have no memory of what this notebook was about, or what this or that command was intended to achieve. Documenting and taking note extensively will look laborious at first, but you will thank yourself in the future when you will come back to that same notebook. Even things that seem obvious (like 'this variable is for XYZ purpose') will be immensely useful, once you will have forgotten the context of the notebook. It is easier to skip some notes that you do not need to read (because you remember enough), than stare for hours to a block which purpose you can't figure out anymore.

Go back to the Hello World block (click inside the block). This time, type Option+Enter (or Alt+Enter). You should see that the command executes again, but also creates a new empty block before your markdown notes. In this empty block, you can insert more notes or run more code without having to move what is below.

Go back to the Hello World block. After the command, add some text after the pound sign, for example:

```shell
print('hello world') # this is just a test
```

Press Shift + Enter. You will see that the note after the pound sign stays visible in the block. This type of in-code annotation is very useful to comment on details, document what a variable does etc. Here again, you should use this extensively.


Now it is time to start working on some code. First, let's import some libraries we will need for our computation:

```shell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
```

If you find that one command returns "XYZ not found" or "no module XYZ" in Jupyter, you have two solutions. From your virtual environment, you can install the missing package. As the process depends on the environment manager, I won't document here all the options, but you start there.
You can also install the missing element from Jupyter, with the !pip command, for example:

```shell
!pip install numpy 
```
The exclamation point is used to tell Jupyter to run this command in the console. The notebook will tell you if the package is already installed. 

![alt test](JL_installs.png?raw=true "Installing packages in Jupyter Labs")

Numpy is a great library to manipulate numbers. As we do not want to call 'numpy this' or 'numpy that' every time we call a numpy function, we tell Python that we will use the shorthand 'np'.

Pandas is great to manipulate tables that will look like excel spreadsheets, allowing table-wise, row-wise or column-wise fast operations.

Pyplot, in the matplotlib set of libraries, is great to plot simple graphs.

Next, you will need data, in this same folder, download the [unconv_MV_v5.csv file](https://github.com/jhenry-github/MLIOT-2024-content/blob/main/TPs/unconv_MV_v5.csv). Modify the path to access the file, I am using below my own settings.

You then want to import the training set as a pandas dataframe (we give it the short name df).

```shell
df = pd.read_csv('/Users/jerhenry/Documents/Perso/IMT/IMT_ML_IoT/unconv_MV_v5.csv');
```

Note the semi-colon sign at the end of the line, that tells Python to complete the command without outputting the result.

When importing data, first good step is to look at the data. When it is a dataframe, looking at the first few lines will give you an idea of the column names, and of the first few values:

```shell
df.head()
```

Try to put a number in the parenthesis, or a range (e.g., 2-5).

Another good way to explore the data is to graph it. Let's design a graph, with the x axis taken from the viscosity, Por, and the y axis taken from the pressure, Prod:

```shell
plt.figure(figsize=(18,12))
plt.plot(df[['Por']], df[['Prod']], 'o')
plt.title("Optimal Pump Pressure Measurements")
plt.xlabel("Viscosity (Por)")
plt.ylabel("Pressure (Prod)")
```

Try to change the numbers in figsize, and change the labels to better undersdtand what these variables do.

Let's move into gradient descent. The first attempt will be to perform the gradient descent manually. We first create a shorthand notation, x for the viscosity, y for the pressure columns. Note that the command achieves two goals: one is to extract the column of the df called Por (and the one called Prod), and another goal is to convert this column, from a df column, into a numpy array, making it effectively a vector:

```shell
x = np.array(df[['Por']])
y = np.array(df[['Prod']])
```

Try to call x, or y, to see what the output looks like. Also try to experiment (for example 'not' converting into an array) to see what the different formats are.

We also want to extract the lentgh of these vectors, so we can run the loop on all 'n' entries in the columns:

```shell
n = len(df[['Por']])
```

Now, to run gradient descent, we need to change the variables theta0 and theta1 until we find convergence. But these variables need to start from some value. So we pick up two initial values for theta0 and theta1. You could use random values, but it is common to start with 0s:

```shell
th0_curr = th1_curr = 0
```

We also need to decide how many times we will try to explore new values, which means how many iterations we get our algorithm before we decide that it should have converged.  will run the loop. It is common to start with something like 500 or 100, then refine later. That second part is important, you will need to adjust, in most cases, after your first attempt.

```shell
iterations = 1000
```

Then, we need to decide by how much we change theta0 and theta1, for now let's use a fixed number, something small. Here again, this is just a first attempt, you will likely have to try a few numbers.

```shell
learning_rate = 0.02
```

It is now time to run gradient descent. We are going to run a loop 'iteration' times (above, 1000). Each time, we are going to pick one value from x (Por), compute with our theta0 and theta1 the expected matching y value (with the operation we saw in class, y_predicted = th1_curr * x + th0_curr), then we are going to compare that predicted y to the real y matching the x value in the Prod column. That difference is the cost (y - y_predicted). We are going to run this operation column-wise, i.e., we do it for all values in x and y, and the cost is the sum of all these differences (thus 'sum' in the dth0 and dth1 below). 

Then, we are going to use this cost to decide how we want to change theta0 and theta1 for the next iteration. We are going to compute the derivative of theta1 and the derivative of theta0. The derivative is the slope, which tells us if the curve at this point is going downward or upward. You may remember from the class that the ideal slope is flat, i.e., slope is 0, which means that at this point the prediction is as close as it can be to the real y value. So if our derivative is negative, the curve is going downward and we want to continue toward the flat zone (so theta0 and theat1 should be positive in next iteration). If the derivative is positive, we are climbing away from the flat area, and we want to go backward (theta1 or theta0, whichever derivative you got positive), should be negative in next round.

Think about it at the comparison level. If the difference between the real y and the predicted y is positive (y is larger than y-predicted), then we need y-predicted to be a bit bigger, so we need theta1 and theta0 to be a bit larger, and vice versa. 

So there are a few ways to achieve this. The derivative gives us a negative number for dth1 and dth0 (go back to the class material if you forgot how it is computed). So, taking dth0 as an example, if dth0 is negative (y is larger than y-predicted), then we want theta0 to be a bit larger, so we compute the next theta0 as the previous theta0 minus dth0 (as dth0 is negative, the new theta0 is bigger than the previous one). In fact, we could do something simpler, which is look at the difference (y minus y_predicted), then add something, for example our learning rate value, if the difference is positive (-> we want to increase y_predicted, so we want to increase theta0). If the difference is negative, remove the value of the learning rate, and change theta0/theta1 that way, at the 'learning rate' pace, for each iteration. We kinda do that, but we want to be a bit more clever. By multiplying the learning rate by the derivative, we make the change big if the derivative is big (i.e., the slope is stiff, and we are far from the 'flat' area). But if the derivative gets small, the change gets smaller too (so we slow down as we approach the flat area of the curve, to make sure 'not' to miss the minimum).

We could run the loop this way, but then we would be missing a lot, as we would only get the result. It is nicer to see what happens, so at each iteration, we are going to compute and display the total cost, so we can track it, and verify that it is going down. It will not get to 0 (as you saw above that the points show a general linear tendency, but they are not all on a single line), but it should get lower from the first to the last iteration. We are also going to dispaly the iteration number, and the values of theta0 and theta1 at this point. Let's do it:

```shell
for i in range(iterations):
    #at each step,  we take the x value, and use our theta0 and theta1 to predict some y value (likely wrong at the beginning)
    y_predicted = th1_curr * x + th0_curr
    #as we need to modify a bit theta0 and theta1 at each step, we calculate (at each step), the derivative of each theta
    dth1 = -(2/n)*sum(x*(y - y_predicted))
    dth0 = -(2/n)*sum(y - y_predicted)
    #then our next theta is going to be changed by the value of the derivative times the learning rate. T
    th1_curr = th1_curr - learning_rate * dth1
    th0_curr = th0_curr - learning_rate * dth0
    # one good way to see what is going on is to print at each iteration the thetas and the cost
    # if everything works well, then the cost should be going down. So we don't need the cost for the loop itself,
    # but we want to compute it here, just so we can print it and see if it is going down:
    cost = (1/n) * sum ([val**2 for val in (y - y_predicted)])
    print("th1 {}, th0 {}, cost {}, iteration {}".format(th1_curr,th0_curr,cost,i))
    
```

Now it is time to experiment a bit. You started with our suggestion above, 1000 iterations and learning rate of 0.02. Try different values, for example 2000 iterations, a learning rate of 0.002, etc. As you experiment, you should find some values where you see that the cost keeps going down all the way to the last iteration (okay, likely faster at the beginning than at the end). If the cost goes down then back up, your learning rate is too high, and you bounced above the minimum. Try a smaller learning rate. At the same time, if the change of cost kind of flattens (not much difference anymore between iterations), then you may have too many iterations. On the other hand, if the cost keeps going down quite much in the last iterations, then you need more iterations... it is a bit of a trial and error game here, and there is no absolute good response, only values that you will end up be satisfied with.

Once you have the numbers you like, let's plot our data again, and overlay there the line you found. There are a few ways to do this. As our x values range from 5 to 25, we can just compute two points on the line, one at x=5.5 and the other at x=24.5 (we compute the predicted y for each, now that we have our thetas). The below code is ugly, but the goal is to show you what happens in a simple way, even if you do not master python:

```shell
# two points, one at x = 5.5, the other at x=24.5, then their matching predicted y, and we draw a line between these points, this is your predicted line
A1 = 5.5
B1 = int((th1_curr * A1 + th0_curr).item())
A2 = 24.5
B2 = int((th1_curr * A2 + th0_curr).item()) #take the last th0_curr/th1_curr in the list

P1 = [A1, A2]
P2 = [B1, B2]
# the same figure as before:
plt.figure(figsize=(18,12))
plt.plot(df[['Por']], df[['Prod']], 'o')
plt.title("Optimal Pump Pressure Measurements")
plt.xlabel("Viscosity (Por)")
plt.ylabel("Pressure (Prod)")
# adding our line:
plt.plot(P1, P2)
```

This was great fun, and you can see that you can use manual gradient descent if you have to. Of course, automating the whole thing would be nice, and this is what scikit learn libraries are for. As we are looking for a line, let's import the linear model from scikit learn, and then let's create a linear regression model (we call this an object).

```shell
from sklearn import linear_model
# let's create a linear regression object
reg = linear_model.LinearRegression()
```

Scikit learn has many models. By calling LinearRegression(), we are loading the model that uses least squares (it also has a few more automated optimizations), which is the technique we ran manually above. But there are other models. Take a bit of time to explore [on their site](https://scikit-learn.org/stable/supervised_learning.html) and see the other models that you could use.

Once we have the model, it knows what we want to do: if we have X and y as above, it needs to find theta0 and theta1 with the same technique we used above. So the only thing we need to ask is to do just that, which is called "find the best fit for theta1 and theta0, given this X and this y". This is done in a single line:

```shell
reg.fit(df[['Por']],df[['Prod']])
```

That's it. In the background, the library will figure out the best number of iterations and the best learning rate, then compute the gradient descent as we did above (without displaying each iteration, because most people only care about the result, not the process).

If you want to see the thetas, theta0 is called the intercept, and theta1 is called the coefficient:

```shell
intercept_scalar = reg.intercept_.item() if hasattr(reg.intercept_, "item") else reg.intercept_
coefficient_scalar = reg.coef_[0]  # Since coef_ is always an array, extract the first element

# Print the values
print(f"Intercept (theta0): {intercept_scalar}")
print(f"Coefficient (theta1): {coefficient_scalar}")
```

Compare these values to the ones you found above. If you did your job well, they shouldn't be too far, the main difference is that the library does it much faster than us, trying manually different values until we find the right one.

Once the computation is done, let's plot the whole thing again:

```shell
# Define the A values (inputs) for which we want predictions (B values)
A_values = [3, 5.5, 24.5]  
B_values = reg.predict(pd.DataFrame(A_values, columns=['Por'])).flatten()  # Predict outputs

# Print the predictions
for A, B in zip(A_values, B_values):
    print(f"Input A: {A}, Predicted B: {B}")

# Plot the data
plt.figure(figsize=(18,12))
plt.plot(df[['Por']], df[['Prod']], 'o', label="Data")
plt.title("Optimal Pump Pressure Measurements")
plt.xlabel("Viscosity (Por)")
plt.ylabel("Pressure (Prod)")

# Plot the predicted points
plt.scatter(A_values, B_values, color='red', label="Predicted Points", zorder=5)

# Draw the regression line (between min and max of 'Por')
x_range = [df['Por'].min(), df['Por'].max()]
y_range = reg.predict(pd.DataFrame(x_range, columns=['Por']))
plt.plot(x_range, y_range, color='green', label="Regression Line")

plt.legend()
plt.show()

```

Compare this line to yours. Here again, it should be fairly close in this simple example.

Last, now that you have a theta0 and theta1, you can run inferences. In other words, if you get a particular Por (x) value, you should be able to predict the most likley Prod (y) value. Of course, you can do it by have with theta0 + theta1 * x, but the scikit learn library has integrated the function, called predict. For example:

```shell
# Suppose a new Por value:
New_Por = 8
#let's predict the presure for that Por:
New_Pred = reg.predict(pd.DataFrame([[New_Por]], columns=['Por'])).item()

# Print the prediction
print(f"The predicted value for Por = {New_Por} is: {New_Pred}")
```

A good part of ML is trying to understand the data. Por is related to Prod, but it is also related to another parameter. Spend some time graphing Por as x against the other columns (as y), you will find one of them that also displays a linear relationship with Por. Besides Por (molecular porosity) and Prod (production output) that you already know, the set includes Perm (permeability, how well water can mix with the oil), AI
(accoustic impedance, how well sound traverses the product), Brittle (brittleness of hard particles), TOC (total organic carbon), and VR (reflectance).

After a while, you may discover a few candidates. For this part, let's use TOC. Let's plot to visually guess if TOC has some relationship with Prod.

```shell
plt.figure(figsize=(18,12))
# A fist step is to look at if POC is also linear with Prod
plt.plot(df[['TOC']], df[['Prod']], 'o')
```

The relationship seems somewhat linear. Then we can create a multivariate linear regression model. I won't have you do it manually, but please ask if you want to try and it does not work. With scikit learn, training a multivariate model is about the same command as a single variate fitting request.

```shell
reg.fit(df[['Por', 'TOC']],df[['Prod']])
```

Done. Now let's look at our coefficients and intercept values.

```shell
# Extract and print intercept and coefficients
intercept_scalar = reg.intercept_.item() if hasattr(reg.intercept_, "item") else reg.intercept_
coefficients = reg.coef_.flatten()  # Flatten to ensure a clean 1D list if needed

# Display results
print(f"Intercept (theta0): {intercept_scalar}")
print("Coefficients:")
for feature, coef in zip(['Por', 'TOC'], coefficients):
    print(f"  Coefficient for {feature}: {coef}")
```

Nice. We can also visualize the line we found. As we are in 3D, the command to plot is a bit more complicated than in 2D. 3D is the max we can plot in this universe, so we are at the top of our game here.

```shell
# syntax for 3-D projection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Set up the 3D plot
fig1 = plt.figure(figsize=(18, 12))
ax = plt.axes(projection='3d')

# Scatter plot of the data
ax.scatter(df['Por'], df['TOC'], df['Prod'], color='blue', label="Data Points")

# Labels and title
ax.set_title('Optimal Pressure Based on Por and TOC')
ax.set_xlabel("Viscosity (Por)")
ax.set_ylabel("Granularity (TOC)")
ax.set_zlabel("Pressure (Prod)")

# Predicting for new points (A5, B5) and (A6, B6)
A5, B5 = 5.5, -0.2
A6, B6 = 24.5, 2.2

# Input must be a DataFrame with correct feature names
C5 = reg.predict(pd.DataFrame([[A5, B5]], columns=['Por', 'TOC'])).item()
C6 = reg.predict(pd.DataFrame([[A6, B6]], columns=['Por', 'TOC'])).item()

# Points for the predicted line
P5 = [A5, A6]  # x-axis values (Por)
P6 = [B5, B6]  # y-axis values (TOC)
P7 = [C5, C6]  # z-axis values (Prod predictions)

# Plot the predicted line
ax.plot(P5, P6, P7, color='red', label="Predicted Line")

# Add legend
ax.legend()

# Show the plot
plt.show()

```

The line is slightly different from the 2D model, but not far.

Last, in real life you will want to save your model, once it is trained, because training is going to take hours, days or weeks, and you will want to reuse this model elsewhere.

There are multiple ways to save a model created with scikit learn, the simple one is to use joblib.


```shell

import joblib
# Save your model to a file - you should see that file in your working directory
joblib.dump(reg, 'my_cool_joblib_model')
```

Of course, you are on a single machine, but you can now take this model and load it somewhere else. So imagine that you are on another machine, and load your model:

```shell
mj = joblib.load('my_cool_joblib_model')
```

Once you model is loaded (we called it mj), you can use it for inference as usual:

```shell
new_input = pd.DataFrame([[5.5, -0.2]], columns=['Por', 'TOC'])

# Predict the output using the trained model
predicted_output = reg.predict(new_input)[0]  # Extract the first element directly

# Print the result
print(f"The predicted output for Por = 5.5 and TOC = -0.2 is: {predicted_output}")
```
