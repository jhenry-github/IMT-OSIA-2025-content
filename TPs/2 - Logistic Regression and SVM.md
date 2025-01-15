# Logistic regression in Jupyter

Logistic regression does not seek to find the equation that best describes your data, but find which data belongs to which group. The csv file contains data useful to our purpose here, namely the record of when the pump ended up being clogged after the pressure was changed. As usual, it is great to visualise data. You may want to go to the first notebook on Linear regression, and get from there the blocks you need to reload the standard libraries (numpy, pyplot, etc.) along with the csv file, as we will need them here as well. Then, let's plot the columns of interest:

```shell
plt.figure(figsize=(18,12))
plt.plot(df[['Brittle']], df[['Reuse']], 'o')
plt.title("Pumps issues based on grains brittleness")
plt.xlabel("Brittleness")
plt.ylabel("1 if pump could be reused, 0 if it was clogged")

```

As you can expect, scikit learn incorporates the libraries for logistic regression. We didn't do this for the previous exercise, but as we go further, we want to incorporate more and more good practices. One of them is to split the dataset into a training set and a test set. You can do it manually, but you would also expect that there is a simple command for that. And there are many, one of them in scikit learn. You use it by defining what is the training percentage (below, 80% for training, 20% for testing), then calling out the names of the training and test parts (for the X and Y values, when you run a single variable case as we do here):

```shell
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df[['Brittle']], df.Reuse, train_size=0.8)

```

Next, you can load the logistic regression library. Before, we called the model 'reg', then 'mj'. The name does not matter, as long as you remember it. Also, in a notebook where you use different models, it is also a good practice to give them different names, so that you know which one you are calling. So let's name this one 'model':

```shell
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

```

Now, just as before, all we have to do is to call the 'fit' command on our data. As we split the data between a training and test sets, we call the fit command on the training part (as the goal, with fit, is to train the model):

```shell
model.fit(X_train.values, Y_train)

```

Now if we used fit, we found coefficients. Let's have a look at them:

```shell
model.intercept_, model.coef_

```

For a logistic regression task, the coefficients may not mean much to you. An easier way is to look at the model performances on the test set. First, have a look at the real data (did the pump get clogged [1] or not [0]):

```shell
Y_test

```

Now compare with the model predictions (for the X_test values). If the model worked, most of the predicted y values should be the same as the real y values you just saw:

```shell
list(model.predict(X_test.values))

```

The output is simplified (0 or 1). In the real world, you may want to see the real prediction, which is a probability value (probability of 0, probability of 1). In this simple case, the values are very close to the simplified version, but at least you can see the real probabilities:

```shell
model.predict_proba(X_test.values)

```

To illustrate how the prediction works by projecting the probability onto a curve, we can generate brittleness values (from 10 to 85, by jumps of 0.5), then plot the prediction:

```shell
brittleness = np.arange(10, 85, 0.5)
probabilities = []

# Ensure input matches feature names
for i in brittleness:
    input_df = pd.DataFrame([[i]], columns=X_train.columns)  # Replace 'X_train.columns' with the actual column names
    p_clogs = model.predict_proba(input_df)
    probabilities.append(p_clogs[:, 1])
plt.scatter(brittleness,probabilities)
plt.title("Logistic Regression Model")
plt.xlabel('Brittleness')
plt.ylabel('Status (0: clogged, 1: reused)')

```

# Support Vector Machines in Jupyter

SVM is the companion to any supervised activity like logistic regression. It allows you to find the boundary between groups, and therefore speed up the recognition of group membership during the inference phase.
Suppose your board was mounted into a necklace (or a belt buckle). Of course, other devices (like smartphones) would be possible too. Then, in the training phase, volunteers were asked to perform specific activities (walking, walking upstairs, walking downstairs, sitting, standing, laying). Each time they were performing one activity,
the sensor would have recorded 3-axial angular acceleration (at a rate of 50 Hz). Each activity would then have been labeled. You can find more details on how the experiements were performed, the number of vounteers, the data filtering technique applied etc.
on the [UCI site](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). The data is downloaded as a set of txt files in  [this zip archive](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip).

First, look at the data, you will see that the folder has subfolders with txt files, following this structure:

```shell
UCI HAR Dataset/
├── activity_labels.txt
├── features.txt
├── train/
│   ├── X_train.txt
│   ├── y_train.txt
│   ├── subject_train.txt
│   └── Inertial Signals/
│       ├── total_acc_x_train.txt
│       ├── total_acc_y_train.txt
│       ├── total_acc_z_train.txt
│       ├── body_acc_x_train.txt
│       ├── body_acc_y_train.txt
│       ├── body_acc_z_train.txt
│       ├── body_gyro_x_train.txt
│       ├── body_gyro_y_train.txt
│       ├── body_gyro_z_train.txt
├── test/
│   ├── X_test.txt
│   ├── y_test.txt
│   ├── subject_test.txt
│   └── Inertial Signals/
│       ├── total_acc_x_test.txt
│       ├── total_acc_y_test.txt
│       ├── total_acc_z_test.txt
│       ├── body_acc_x_test.txt
│       ├── body_acc_y_test.txt
│       ├── body_acc_z_test.txt
│       ├── body_gyro_x_test.txt
│       ├── body_gyro_y_test.txt
│       ├── body_gyro_z_test.txt
```

The structure is quite obvious, labels tell you the names of the activity bits, train and test are the training and test sets, already created for you, and intertial signals the (x,y,z) movements of the sensors. Subjects are the volunteers identifiers.
The first step is therefore to load this data. One small difficulty is that there are many files. Another difficulty is that many of the files are of the same type, and therefore many files have columns that have the same name. So a first step is to load the name of the features, then create a panda data frame that is going to include all files, one after the other, with the right data and under the right feature name.

```shell
#Let's load the dataset

import pandas as pd

# Define dataset path, change this to poin tot where the folder is located on your machine.
dataset_dir = '/Users/jerhenry/Documents/Workdoc/f-Virtual_Machines/Classes/IMT24-25/human+activity+recognition+using+smartphones/UCI_HAR_Dataset/'

# Load feature names
features = pd.read_csv(f'{dataset_dir}/features.txt', sep='\s+', header=None, names=['Index', 'Feature'])

# Enforce uniqueness of feature names by appending unique suffixes, we need this phase because the dataset has a bunch of files with the same column names
features['Feature'] = features['Feature'].apply(lambda x: x.strip())  # Strip any whitespace
unique_feature_names = pd.Series(features['Feature']).astype(str)
unique_feature_names = unique_feature_names + "_" + unique_feature_names.groupby(unique_feature_names).cumcount().astype(str)
print("Unique feature names generated successfully!")

# Convert to list
unique_feature_names = unique_feature_names.tolist()

```
The original data set already has separated the training and the test data. This is great and we want to load both of them. However, we want to run a labeling simplification operation first. Therefore, We  want to merge the training and the test data into one large data set, and then do the split when needed.

```shell
# Load training data
X_train = pd.read_csv(f'{dataset_dir}/train/X_train.txt', sep='\s+', header=None, names=unique_feature_names)
y_train = pd.read_csv(f'{dataset_dir}/train/y_train.txt', sep='\s+', header=None, names=['Activity'])
subject_train = pd.read_csv(f'{dataset_dir}/train/subject_train.txt', sep='\s+', header=None, names=['Subject'])

# Load testing data
X_test = pd.read_csv(f'{dataset_dir}/test/X_test.txt', sep='\s+', header=None, names=unique_feature_names)
y_test = pd.read_csv(f'{dataset_dir}/test/y_test.txt', sep='\s+', header=None, names=['Activity'])
subject_test = pd.read_csv(f'{dataset_dir}/test/subject_test.txt', sep='\s+', header=None, names=['Subject'])

# Combine training and testing datasets
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
subjects = pd.concat([subject_train, subject_test], axis=0).reset_index(drop=True)

# Combine everything into a single DataFrame
data = pd.concat([subjects, y, X], axis=1)

# Map activity IDs to labels
activity_labels = pd.read_csv(f'{dataset_dir}/activity_labels.txt', sep='\s+', header=None, names=['ID', 'Activity'])
data['Activity'] = data['Activity'].map(activity_labels.set_index('ID')['Activity'])

# Display the combined dataset
print("Dataset loaded successfully!")
print(data.head())

```

The command data.head() allows us to display the first few rows of our dataset. The numbers may not tell you much. However, you should be able to see that some activity names are associated with measurements. The measurement names may be clear enough. For example, tBodyAcc-mean is the (triaxial) acceleration of the body, averaged over the sampling period. Refer to the UCI URL if you want the details of each column.

Our next step is to separate the measurement data from the labels. The labels will be our target y, and the activities will be our multidimensional input dataset X. We then want to convert values that are strings into integers, because SVM will try to find the boundaries between numbered groups (groups 1, 2, etc.), so it does not work well when the groups are names instead of numbers (“walking”, “sitting”, etc.)
The  numbers are for SVM, but we (humans) still want to eb able to read label names.

```shell
# Extract features (X) and labels (y)
X = data.drop(columns=['Subject', 'Activity'])  # All features except identifiers
y = data['Activity']  # Target labels (activity names)

# Encode labels to integers (SVM requires numerical targets)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)

```
Now we can split the data set back into the training and the testing parts. If you looked at the numbers in the data.head() command, You probably saw that the scale of these numbers seems to be between -1 and +1. This is great, but we cannot be sure. Therefore, a good habit is to always use the standard scaler to rescale the data to the same range. If the data was already in the same range, then you do not lose anything. But if the data was on different ranges, then you remove the bias that may exist when one range is much larger than the others.

We can then verify that the training and test sets have the same dimensions, and we can also look at their respective sizes.

```shell
#Assign the pre-split data  to "train" and "test".
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# We also need to scale the training and test sets, as the scale of the values are not the same for all measured values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Use training set to fit the scaler
X_test_scaled = scaler.transform(X_test)        # Scale the test set with the same scaler

# Verify shapes
print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)

```

Then we can go on to train the SVM model. For this task, we define the y_train and the y_test sets, and then we use scikit Learn integrated SVM tool for the training. The rbf part says that we are using the Radial Basis Function to measure the similarity between two points, and therefore decide if they are in the same group or not. Then gamma controls the influence of individual training samples on the decision boundary. Last, C, the regularization parameter, controls the tradeoff between maximizing the margin of the decision boundary and minimizing classification errors. You may remember from the class part that this parameters allows you to have outliers in some groups.

Once we have the model trained, we can use it to predict the labels on the test set. Then we can ask the algorithm to display the accuracy of our prediction when we apply it to the test set.


```shell
# Initialize and train the SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Flatten target labels
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Initialize and train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


```

Our boundaries seem to be pretty good, because our precision is around 0.95. The classification report gives u a precision value for each of the categories, but also a recall value. The precision is the proportion of correct predictions, in the form (true positives)/(true positives + false positives). 
High precision means fewer false positives (the model rarely misclassifies a class as positive when it is not). Recall measures how many of the actual positive instances were correctly predicted. It is computed as (true positives)/(true positives + false negatives). 
High recall means fewer false negatives (the model rarely misses actual positive instances). 
The f1-score balances the two metrics. Formally, it is the harmonic mean of Precision and Recall, calculated as 2x (Precision x Recall)/(Precision + Recall). T
hen, Support is the number of true instances in each class (for example, the Laying class had 496 samples in the dataset, irrespective of what our SVM algorithm predicted). It is useful, because if you have a class with low support, then there are not many samples to train from, which may explain why, for that class, the precision or recall may be lower than for the others. In our case, the classes are well balanced.

These numbers are interesting, but they are sometimes difficult to consume in a column form. Another way to display these performances is to build a confusion matrix. The confusion matrix shows for each class how many samples were correctly labeled and how many were incorrectly labeled. The matrix allows you to look at each class and see against which other class it performs well and against which other class it performs poorly.


```shell
# Compute and plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVM Activity Recognition')
plt.show()

```

Overall, our SVM algorithm seems to perform well. It's only for the Walking category where sometimes the algorithm confuses walking downstairs and walking straight forward.

With these kind of classification, it is very tempting to want to do what we did in the previous labs where we wanted to plot our points so that we could see clusters and maybe see the boundaries between these various categories. 
In this case, however, this activity would be a little bit difficult because the data set has 561 parameters and that means that we would need to plot in 561 dimensions, which may be a little bit difficult to represent in 2 dimensions. So what we can do in order to be able to plot some groups? 
One option is to reduce the dimensions with the PCA algorithm to two dimensions. Of course, doing so will dramatically simplify the data, so we will lose a lot of information about our various groups. But maybe we will be able to see some pattern that will allow us to understand what the differences between these different groups are.

To perform that task, we are going to first load the data set, then apply the PCA algorithm to the data set so as to retain only two dimensions. Then we will train SVM again, this time on the bi dimensional data set. And then we'll be able to plot the points we will have found.


```shell
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Step 1: Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 2: Train SVM on 2D PCA-reduced data
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_pca.fit(X_train_pca, y_train)

# Step 3: Create a mesh grid for visualization
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict on the mesh grid to get decision boundaries
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Step 4: Plot decision boundaries and training points
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.viridis, edgecolors='k')
plt.title("SVM Decision Boundaries on PCA-Reduced Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Activity Classes')
plt.show()
```

Take some time to try to understand the plot that we have obtained. The colors represent the different categories from 1 to 6. It is fairly obvious that categories yellow and green, that is, five and six, seem to form a specific group, while the other categories 1, 2, 3, and maybe 4, seem to form another group. It could be interesting to try to understand what these activities are, in other words, which activity contributes to which principal component.

To find these components memberships, we can load PCA again and try to create a data frame where each principal component is a column, then try to associate each feature to its primary column. This way we will be able to see which feature belongs to which principal component.


```shell
# First we insert all parameters into PCA
pca_full = PCA(n_components=2)
X_train_pca_full = pca_full.fit_transform(X_train_scaled)

# Get the PCA loadings
pca_loadings = pca_full.components_

# Create a DataFrame to associate features with their contributions
feature_contributions = pd.DataFrame(pca_loadings.T, 
                                     columns=['PC1', 'PC2'], 
                                     index=X.columns)

# Sort the features by their absolute contribution to each component
top_features_pc1 = feature_contributions['PC1'].abs().sort_values(ascending=False).head(5)
top_features_pc2 = feature_contributions['PC2'].abs().sort_values(ascending=False).head(5)

# Display the top features for each principal component
print("Top contributing features to Principal Component 1:")
print(top_features_pc1)

print("\nTop contributing features to Principal Component 2:")
print(top_features_pc2)
```

Now you can see which features contribute primarily to principal component 1 and which features contribute primarily to Principal component 2. This column form presentation is not necessarily very easy to read, so maybe we can do better, and have a bar graph of these two principal components and their contributors.



```shell
import matplotlib.pyplot as plt

# Plot top features for PC1
plt.figure(figsize=(8, 5))
top_features_pc1.plot(kind='bar')
plt.title('Top Features Contributing to Principal Component 1')
plt.xlabel('Features')
plt.ylabel('Contribution')
plt.show()

# Plot top features for PC2
plt.figure(figsize=(8, 5))
top_features_pc2.plot(kind='bar', color='orange')
plt.title('Top Features Contributing to Principal Component 2')
plt.xlabel('Features')
plt.ylabel('Contribution')
plt.show()

```

When looking at these different contributions, your intuition may tell you that there seems to be some pattern. 
It seems that components that represent certain acceleration, like jerking or turning, seem to be heavy contributors to principal component 1. 
On the other hand, components that seem to represent mean values seem to contribute more to principal component 2. 
In other words, more sudden changes seem to have some importance for component 1, but then the continuity of the changes seems to have a lot of importance for component 2. 
Let's see if we can understand this logic a little bit more so that we understand our data better.

If we look at our categories a little bit closer. And we try to find. Two big categories. We see that maybe we can group all activities that are related to continuous moving actions like walking or going up or down a stair, and all activities that represent static  “one shot” actions like sitting, standing or laying down.
So let's bundle the activities in that way and try to run SVM again to see if we can find something useful. The code here is about the same as the one we used above, with just the initial action of grouping all activities of the same type into the same label. The rest is the same SVM as above.


```shell
# The general process is the same, we just group activities
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Relabel the "Activity" column into 2 classes
# Define the moving and not-moving activities
moving_activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']  # Add more if needed
not_moving_activities = ['STANDING', 'SITTING', 'LAYING']

# Create a binary label column
data['BinaryActivity'] = data['Activity'].apply(lambda x: 1 if x in moving_activities else 0)

# Step 2: Extract features and the new binary labels
X = data.drop(columns=['Subject', 'Activity', 'BinaryActivity'])  # Features
y = data['BinaryActivity']  # Binary labels

# Step 3: Split into training and test sets (already pre-split in HAR dataset)
X_train_scaled = scaler.fit_transform(X_train)  # Assuming X_train and X_test are already loaded
X_test_scaled = scaler.transform(X_test)

# Ensure binary labels for train and test
y_train_binary = y[:X_train.shape[0]].values.ravel()  # Use first N rows for training
y_test_binary = y[X_train.shape[0]:].values.ravel()  # Use remaining rows for testing

# Step 4: Train the SVM model
svm_binary = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_binary.fit(X_train_scaled, y_train_binary)

# Step 5: Predict on test set
y_pred_binary = svm_binary.predict(X_test_scaled)

# Step 6: Evaluate performance
print("Accuracy on binary classification (Moving vs Not Moving):", accuracy_score(y_test_binary, y_pred_binary))
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, target_names=["Not Moving", "Moving"]))

# Step 7: Confusion Matrix
conf_matrix_binary = confusion_matrix(y_test_binary, y_pred_binary)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_binary, annot=True, fmt='d', cmap="Blues", xticklabels=["Not Moving", "Moving"], yticklabels=["Not Moving", "Moving"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Moving vs Not Moving')
plt.show()

```
This is interesting. This time our precision is very high. Let's try to run PCA again on these two-class structure and see if we can plot again the different components. 
We still have 561 dimensions, but because we have two categories only this time we may be able to have a better grasp into which category represents which principal component. 
Here again the code is the same as above. We first run PCA to reduce the features to two dimensions. Then we run SVM again on this two-dimensional data set, then we plot the different categories and the decision boundaries.


```shell
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Perform PCA to reduce features to 2 components
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 2: Train the SVM model on PCA-reduced data
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_pca.fit(X_train_pca, y_train_binary)

# Step 3: Visualize the decision boundaries
# Create a mesh grid for plotting
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict the class for each point in the mesh grid
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Step 4: Plot the decision boundaries and the data points
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_binary, cmap=plt.cm.coolwarm, edgecolors='k', label="Training Points")
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_binary, cmap=plt.cm.coolwarm, marker="x", label="Test Points")
plt.title("SVM Decision Boundaries (PCA-Reduced Data: Moving vs Not Moving)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(loc='upper left')
plt.colorbar(label="Classes: 0 (Not Moving), 1 (Moving)")
plt.show()

# Step 5: Show the most important features contributing to PCA
pca_loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=X.columns)
top_features_pc1 = pca_loadings['PC1'].abs().sort_values(ascending=False).head(5)
top_features_pc2 = pca_loadings['PC2'].abs().sort_values(ascending=False).head(5)

print("Top contributing features to Principal Component 1:")
print(top_features_pc1)

print("\nTop contributing features to Principal Component 2:")
print(top_features_pc2)

```

Our intuition proves to be right. There are two categories, moving and not moving, and they are very well clustered. 
Any component that represents a sudden change in state like jerking or a rotation contribute to component 1 to define a moving action. Any component that represents a continuity of movement or an amplitude contributes to component 2. 
Component 2 alone would not be sufficient to distinguish moving from non-moving actions, because walking up stairs may represent the same amplitude as standing up for example. But combined with the first component, the classification becomes clear. 
In our embedded system context of course, you can run this inference on the board. Now that you have an SVM model, you can collect data from the sensor, determine if the movement belongs to one group or another, then take action. 
For example, in a hospital, you may want to send an alarm if a patient is suddenly standing up. When monitoring in-home patients, you may want to activate a camera or some audio contact if a person is suddenly laying down then not moving at times of the day where they are not expected to take a nap. 
In this part of the class, we are interested in the machine learning part.












