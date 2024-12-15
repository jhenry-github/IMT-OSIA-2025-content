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

We want to find the zones with the same dew points. For now, let's decide that we have 6 clusters, i.e. 6 different zones, and let's train a k-means algorithm for k=6

```shell
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, verbose=1)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
```







```shell
print('hello world') # this is just a test
```
