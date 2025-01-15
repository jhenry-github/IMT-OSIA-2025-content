In this lab, we want to explore unsupervised learning, namely K-means and DBSCAN. Most of the time you will use K-means, but you will see that there are some cases where DBSCAN is better. 
However, keep in mind that we are often working in multiple dimensions and the projection in in 2 dimensions is sometimes misleading. Let's explore further.
First, let’s load some libraries. In this lab, like in the other ones, you will see that we tend to load the same libraries multiple times. In a real notebook you would not do that, but here I am expecting that you may want to explore some blocks independently, so I want to make sure that you have the libraries you need each time. 

```shell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
```

Next, let's import our training set. In your case, the path will be different, modify to match your path to California_dew_point.csv. 

```shell
df = pd.read_csv('/Users/jerhenry/Documents/Perso/IMT/IMT_ML_IoT/California_dew_point.csv')
```

Let's look at our data. It is a collection of pressure values and dew points for specific locations. This data has been collected with a sensor for atmospheric conditions.


```shell
df.head()
```

Suppose that you are interested in the dew point value for these different regions. You may be wondering if there are some commonalities of dew points from one zone to the other.
 Therefore, you could be tempted to run a 3-dimensional clustering exercise, where you try to plot dew point, latitude and longitude and see if you can form clusters. 
Of course, points near each other (nearby latitude and longitude values) will likely cluster together. 
But will they cluster if you add the dew point value? 
In other words, are there dew point values that can span across a region? 
You would have such clustering for example if nearby areas are at about the same altitude and with the same wind and other climate conditions. Let's try to find out. 
This could be useful, because finding the dew point for one location can allow you to anticipate moving climate conditions (like clouds pushed by the wind), and identify microclimates that may not align with traditional climate classifications.
In turn, this information provides better insights for agriculture, energy planning, or environmental monitoring. For example, your sensor trained on these clusters can predict which humidity zone it currently operates in based on local dew point readings.
The sensor can then anticipate and send commands to initiate or stop spinklers, or detect deviations (fire detection etc.)


The data we care about are of course the dewpoint value, for a specific location, so let's load that into a pandas dataframe.

```shell
X = df.loc[:, ["Dewpoint", "Latitude", "Longitude"]]
```

We want to find the zones with the same dew points. For now, let's decide that we have 6 clusters, i.e. 6 different zones, and let's train a k-means algorithm for k=6:

```shell
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, verbose=1)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
```
With scikit learn, you job is done. This time, I did not fix the random seed, so any new iteration you run of the block above will start from new random values for the cluster center positions. 
Try a few times to verify that you get convergence after different attempts each time. You can see that k-means defines a tolerance value. When the cluster center is set to move for less than the tolerance value, from iteration to the next, k-means declares that the change is insignificant and that the clusters have converged. k-means gives you the expected shift value for these cluster center positions and, as the shift is smaller than the tolerance, k-means declares convergence.
It could be useful to see k-means in action, i.e. watch the cluster centers be placed on the graph, then see them move at each iteration. let's do that. The code below is a bit beyond what I would expect you to learn in this crash course class, but it is commented, in case you are interested in each part. 

```shell
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from sklearn.cluster import KMeans

#  "Latitude" and "Longitude" to numpy
data = df.loc[:, ["Latitude", "Longitude"]].to_numpy()

# Parameters
n_clusters = 6
max_iterations = 300

# Custom KMeans Loop to Track Centroids
def run_kmeans_stepwise(data, n_clusters, max_iterations):
#    np.random.seed(42) #uncomment this if you want consistent results (same number of iterations each time)
    # Randomly initialize cluster centers
    initial_centers = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

    centroids = [initial_centers]  # Store centroids at each step
    current_centers = initial_centers.copy()

    for i in range(max_iterations):
        # Assign points to the nearest cluster center
        labels = np.argmin(np.linalg.norm(data[:, None] - current_centers[None, :], axis=2), axis=1)
        
        # Update cluster centers
        new_centers = np.array([data[labels == j].mean(axis=0) for j in range(n_clusters)])
        
        # Break if centroids do not change significantly
        if np.allclose(current_centers, new_centers, atol=1e-4):
            print(f"KMeans converged at iteration {i + 1}")
            break
        
        centroids.append(new_centers)
        current_centers = new_centers
    
    return centroids, labels

# Run KMeans stepwise
centroid_history, final_labels = run_kmeans_stepwise(data, n_clusters, max_iterations)
n_iterations = len(centroid_history)

print(f"Total Iterations: {n_iterations}")

# Initialize Plot
fig, ax = plt.subplots(figsize=(7, 7))
colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan']

def animate(i):
    """
    Update the animation for the ith iteration.
    """
    ax.clear()
    labels = np.argmin(np.linalg.norm(data[:, None] - centroid_history[i][None, :], axis=2), axis=1)

    # Plot points
    for cluster in range(n_clusters):
        ax.scatter(
            data[labels == cluster][:, 0], data[labels == cluster][:, 1],
            s=5, c=colors[cluster], label=f'Cluster {cluster + 1}'
        )

    # Plot centroids
    for center in centroid_history[i]:
        ax.scatter(center[0], center[1], c='black', s=100, marker='X')

    ax.set_title(f"Iteration: {i + 1}/{n_iterations}")
    ax.legend()
    ax.grid(True)

# Create Animation
ani = animation.FuncAnimation(fig, animate, frames=n_iterations, interval=500, repeat=False)
plt.close(fig)
HTML(ani.to_jshtml())

```

As you run this code,  you should see a play button below the graph. CLick play to wathc the animation. Here again, the random seed value is not there (I commented the line), so you should get a different run each time. Also note that the run for this block is a new run (not the same run as the k-means training we ran in the previous block, we run here a new training and record it so we can play it back - this allows you to get a simple k-means block if all you want is the training result, and/or a block with training and animated graph).

You can also look at X, to check that we have now each point declared to be member of a cluster.

```shell
X.head(12)
```

We started with 6 clusters, but this was a bit of a random choice. What is the right number? Difficult to say, as this is at the scale of the entire California state, it could be hundreds. But let's bet that there may be some macro-climate zones, that we want to discover. If you want to discover micro-climates, feel free to improve the code below. You will see that we bet on up to 10 zones, but you could extend the range much more, to discover these micro zones.
So let's import the silhouette library, and define our possible cluster range.

```shell
from sklearn.metrics import silhouette_score
n_clusters = [2,3,4,5,6,7,8,9,10] # number of clusters
```

Then let's compute the scores. Refer to the class slides if you forgot how these scores are computed. We compute here both the silhouette score and the elbow SSE for each possibility, from 1 to 10 clusters. So for each cluster count target (from 1 to 10), we run k-means and compute the silhouette/elbow scores.


```shell
clusters_inertia = [] # inertia of clusters
s_scores = [] # silhouette scores
for n in n_clusters:
    KM_est = KMeans(n_clusters=n, init='k-means++').fit(X)
    clusters_inertia.append(KM_est.inertia_)    # data for the elbow method
    silhouette_avg = silhouette_score(X, KM_est.labels_)
    s_scores.append(silhouette_avg) # data for the silhouette score method
```

Once we have these scores for all these cluster possibilities, we just need to graph the result, in search for an elbow and a peak in the silhouette method. Let's start with the elbow. As this is a class, I also added a line at the numbers where I saw an elbow. Please check the code, the line is something added manually, not a magical finding from the algorithm. In real life, you would have no line.

```shell
import seaborn as sns
fig, ax = plt.subplots(figsize=(12, 5))

# Use keyword arguments x= and y= for sns.lineplot
sns.lineplot(x=n_clusters, y=clusters_inertia, marker='o', ax=ax)
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Clusters Inertia")

# Optional: Add vertical lines for significant cluster counts
ax.axvline(3, ls="--", c="red", label="Optimal Clusters")
ax.axvline(5, ls="--", c="red")

plt.grid()
plt.legend()
plt.show()
```

As you may see, I saw an elbow at 3, and another at 5. Again, we stop at 10, so there may be other elbows if we go more granular. When you run the code, you may see elbows at other values.

What about with silhouette?

```shell
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x=n_clusters, y=s_scores, marker='o', ax=ax)
ax.set_title("Silhouette Score Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score")

# Optional: Add vertical lines for significant cluster counts
ax.axvline(5, ls="--", c="red", label="Optimal Clusters")
ax.axvline(7, ls="--", c="red")

plt.grid()
plt.show()
```

You can see that I spotted a peak at 5, and another at 7. Both the elbow and the silhouette methods agree for a number of 5, so let's pick that number. By the way, it is fairly common to see a peak in the silhouette method that the elbow method does not see and vice versa. This is why we like to run both, in the hope that they will agree on some numbers. In all cases, you are supposed to have a field expert with you to say whther the number suggested by the algorithms make sense for the data or not.

Now that we have selected 5 as a good candidate, let's run k-means again, with 5 clusters. Here we just run the code, not the animation, but you can reuse the code above if you also want to plot the animation.

```shell
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, verbose=1)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
```

Let's look at the cloud of points and check where these clusters are.

```shell
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Extract only Latitude and Longitude from the centroids
centroids_lat_lon = centroids[:, [1, 2]]  # Assuming Latitude is column 1 and Longitude is column 2

# Plot the clusters
fig, ax = plt.subplots(figsize=(8, 8))

# Plot each cluster with a different color
sns.scatterplot(
    x=X["Latitude"], y=X["Longitude"], hue=labels, palette="tab10", s=50, ax=ax, legend="full"
)

# Plot the centroids on the same plot
ax.scatter(
    centroids_lat_lon[:, 0], centroids_lat_lon[:, 1], 
    c='black', s=200, marker='X', label='Centroids'
)

# Add labels and title
ax.set_title("KMeans Clustering with 5 Clusters (Final Centroids)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.legend()

plt.grid()
plt.show()

```

When you look at the plot that way, you may think that k-means did not do a good job. It may look like some cluster centers are very close from each other. In short, the cluster centers may not look to be at the spot where you would have put them, if you had run the process by hand. This is normal. In fact, our clustering job was on 3 dimensions, latitude, longitude and dew point, but we are plotting in 2 dimensions, namely latitude and longitude. Plotting that way may sound logical because we somehow expect a map of California, but by doing so we are missing a dimension. You could be tempted to plot in 2D, and you would get the following result.

```shell
# Extract only Latitude and Longitude columns
X_2d = df.loc[:, ["Latitude", "Longitude"]]

# Fit KMeans for 5 clusters
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(X_2d)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the clusters
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot for the data points colored by cluster
sns.scatterplot(
    x=X_2d["Latitude"], 
    y=X_2d["Longitude"], 
    hue=labels, 
    palette="tab10", 
    s=50, 
    ax=ax, 
    legend="full"
)

# Plot the centroids
ax.scatter(
    centroids[:, 0], centroids[:, 1], 
    c='black', s=200, marker='X', label='Centroids'
)

# Add labels and title
ax.set_title("KMeans Clustering with 5 Clusters (Latitude vs Longitude)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.legend()

plt.grid()
plt.show()

```


However, you are not considering the dew points, you are just plotting the 2D map of the sensors in California, with special consideration given to the areas where you have more sensors than others. But then you lost the dewpoint information. A better approach in this case is to plot in 3D. We can afford to run this 3D plot, because we are considering 3 dimentions (lat/long, dewpoint). Naturallly, if your data had more dimensions, you would need a different approach. 
The dew points are interesting, because they may connect points that are not exactly near each other, but along the same wind stream and at the same altitude. There are also some zones with a much higher density of sensors than others, which may then output more variations. In short, by plotting in 2D, we are deluding ourselves. One possible fix is to plot in 3D.


```shell
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X["Latitude"], X["Longitude"], X["Dewpoint"], c=kmeans.labels_, cmap='tab10')
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Dewpoint")
plt.show()
```

With this projection, you see the differences in density, but you also get a better view of the groups. In a real use case, you would find a way to rotate the graph so you could look under all the angles, if such rotation would help you understand the data. But already from this default projection, you can see a structure in the data.


So far, k-measn seems to work reasonably well. You may however be tempted to also try DBSCAN, just because we mentioned it in the class. The first step is of course to import the libraries you need, which include DBSCAN from the scikit learn library, but also the Standard Scaler. You may have remember it, as we used it before. Its role is to ensure that all elements (lat/long/dew point) use the same scale, to avoid that one dimensions pulls too much on the others and introduces a bias.


```shell
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Dewpoint", "Latitude", "Longitude"]])

```

Then, as is now usual with scikit learn and its defaults, running DBSCAN is just about calling the fit command.


```shell
# Step 2: Run DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
```

Once DBSCAN calculations complete, we can assign labels to each cluster (here, they will just be cluster numbers), then plot all these clusters.

```shell
# Step 3: Add labels to the original DataFrame for visualization
df["DBSCAN_Labels"] = labels

# Step 4: Plot the DBSCAN results (Latitude vs Longitude)
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot: color points based on their DBSCAN labels
sns.scatterplot(
    x=df["Latitude"], 
    y=df["Longitude"], 
    hue=df["DBSCAN_Labels"], 
    palette="tab10", 
    s=50, 
    ax=ax, 
    legend="full"
)

# Add titles and labels
ax.set_title("DBSCAN Clustering (Latitude vs Longitude)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
plt.grid()
plt.show()

```

If your output is like mine, you may not be super impressed. It seems that DBSCAN found 6 clusters, but the majority of California is part of a single, large cluster. You may think that this is because we are plotting again in 2D instead of 3D. Okay, it's a good possibility, so let's plot in 3D instead.


```shell
# Import libraries
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Dewpoint", "Latitude", "Longitude"]])

# Step 2: Run DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Step 3: Add labels to the original DataFrame
df["DBSCAN_Labels"] = labels

# Step 4: Create the static 3D scatter plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot: color points based on DBSCAN labels
ax.scatter(
    df["Latitude"], 
    df["Longitude"], 
    df["Dewpoint"], 
    c=df["DBSCAN_Labels"], 
    cmap="tab10", 
    s=50
)

# Set axis labels and title
ax.set_title("DBSCAN Clustering (Latitude, Longitude, Dewpoint)")
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Dewpoint")

# Show the static 3D plot
plt.show()

```

Well, if your proejction is like mine, this is still not very impressive. There is one gigantic cluster, and then 5 other small clusters. The reason why DBSCAN fails in this case is because of the two values we have to define, eps, which is the minumum radius of a cluster, and min_samples, which is the minimum number of points needed for a cluster. A good part of the job with DBSCAN is to find the right value for eps and for min_samples. In the code blocks above, (or copy paste in a new block), try to change eps and min_samples to see if you can get something that resembles the 'good' answer provided ny k-means.

In the end, you may find that DBSCAN is not a better option for this case. In fact, the issue is that the data is not particularly noisy, in other words, you cannot say with certainty that there are points in one cluster that should be in the other cluster, there fore the output from k-means is just as good as anything else.

IN fact, as we shared in class, k-means is often the first and best answer. But there are definitely some cases where k-means fails and DBSCAN is better. let's consider a last example. We took your sensor and atatched it to a robotic arm, then recorded the IMU data while the arm was moving up then down toward the right, then down then up towward the left. We stored these various positions in the file [cluster_robotic_arm](https://github.com/jhenry-github/IMT-OSIA-2025-content/blob/main/TPs/cluster_robotic_arms.csv). In the code below, change the path to where you stored the file. The position of the sensor can be visualized as follows.


```shell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Data
# Replace the path with the correct location of robotic_arms.csv
file_path = "/Users/jerhenry/Documents/Perso/IMT/IMT_ML_IoT/2024-2025/Github/cluster_robotic_arms.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the data
print("Data Preview:")
print(data.head())

# Extract features (X_1 and X_2)
X = data[['X_1', 'X_2']].values

# Plot the Original Data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', s=50)
plt.title("Original Data")
plt.xlabel("X_1")
plt.ylabel("X_2")
plt.grid()
plt.show()
```

It is fairly obvious from the recorded positions that there are two gestures there. So one obvious thing we may ask the algorithm is to simply group the points matching each gesture to its own cluster, so you can separate the right-up-right-down gesture from the left-down-left-up gesture. But if you try with k-means, you will get something silly. Let's compare k-means and DBSCAN side by side:

```shell
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Preprocess the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # Expecting 2 clusters
kmeans_labels = kmeans.fit_predict(X_scaled)

# Apply DBSCAN Clustering
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Visualize the Results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# KMeans Clustering
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, label="Centroids")
axes[0].set_title("KMeans Clustering")
axes[0].set_xlabel("X_1")
axes[0].set_ylabel("X_2")
axes[0].legend()

# DBSCAN Clustering
axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='plasma', s=50)
axes[1].set_title("DBSCAN Clustering")
axes[1].set_xlabel("X_1")
axes[1].set_ylabel("X_2")

plt.tight_layout()
plt.show()

# Print Cluster Label Counts
print("KMeans Cluster Labels Count:")
print(pd.Series(kmeans_labels).value_counts())

print("DBSCAN Cluster Labels Count:")
print(pd.Series(dbscan_labels).value_counts())
```

In this case, it is pretty obvious that DBSCAN is the winner. DBSCAN usually wins when data is noisy and you see where the clusters 'should' be. But k-means wins in most of the other cases.



