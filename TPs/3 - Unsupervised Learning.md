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

When you look at the plot that way, you may find that k-means did not do a good job. It may look like some cluster centers are very close from each other. In short, the clusters may not look to be at the spot where you would have put them, if you had run the process by hand. This is normal. In fact, our clustering job was on 3 dimensions, latitude, longitude and dew point, but we are plotting in 2 dimensions, namely latitude and longitude. Plotting that way may sound logical because we somehow expect a map of California, but by doing so we are missing a dimension. The dew points may connect points that are not exactly near each other, but along the same wind stream and at the same altitude. There are also some zones with a much higher density of sensors than others, which may then output more variations. In short, by plotting in 2D, we are deluding ourselves. One possible fix is to plot in 3D.


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

With this projection, you see the differences in density, but you also get a better view of the groups. 




```shell
print('hello world') # this is just a test
```








```shell
print('hello world') # this is just a test
```
