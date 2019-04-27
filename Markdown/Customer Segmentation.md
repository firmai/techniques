
The k-means clustering algorithm works by finding like groups based on Euclidean distance, a measure of distance or similarity. The practitioner selects k groups to cluster, and the algorithm finds the best centroids for the k groups. The practitioner can then use those groups to determine which factors group members relate. For customers, these would be their buying preferences. Clustering is nothing but automated groupbys. With some effort you can create better clusters manually, but you can also just let the data guide you. 


```python
import pandas as pd
customers = pd.read_excel("data/bikeshops.xlsx", sheet=1)
products = pd.read_excel("data/bikes.xlsx", sheet=1)
orders = pd.read_excel("data/orders.xlsx", sheet=1)
```


```python
df = pd.merge(orders, customers, left_on="customer.id", right_on="bikeshop.id")

df = pd.merge(df, products, left_on="product.id", right_on="bike.id")
```

Now the data frame that simulates output we would get from an SQL query of a sales orders database / ERP system

Around here is where you should formulat a question. I would want to investigate the type of customers interested in Cannondale. I apriori believe that Cannondale customers put function of form, they like durable products and that they are after a strong roadbike at a reasonable price. Now we have to think of a unit to cluster on. I think quantity is foundatation and easily interpretable, so I will cluster on that. Something like value is both the function of quantity and price so you would.'t want to cluster on that. Maybe avervage price as it ignores or dampen the effect of quantity. 

The bike shop is the customer. A hypothesis was formed that bike shops purchase bikes based on bike features such as unit price (high end vs affordable), primary category (Mountain vs Road), frame (aluminum vs carbon), etc. The sales orders were combined with the customer and product information and grouped to form a matrix of sales by model and customer. 


```python
df = df.T.drop_duplicates().T
```


```python
df["price.extended"] = df["price"] * df["quantity"]
```


```python
df = df[["order.date", "order.id", "order.line", "bikeshop.name", "model",
         "quantity", "price", "price.extended", "category1", "category2", "frame"]]
```


```python
df = df.sort_values(["order.id","order.line"])
```


```python
df = df.fillna(value=0)
```


```python
df = df.reset_index(drop=True)
```


```python
## You can easily melt which seems to be anothre
```


```python
 #melt()
```


```python
## I think melt reverses the pivot_table. 
## summarise in R is arg, spread is pivot_table/melt
```


```python
df["price"] = pd.qcut(df["price"],2)
```


```python
merger = df.copy()
```


```python
df = df.groupby(["bikeshop.name", "model", "category1", "category2", "frame", "price"]).agg({"quantity":"sum"}).reset_index().pivot_table(index="model", columns="bikeshop.name",values="quantity").reset_index().reset_index(drop=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>bikeshop.name</th>
      <th>model</th>
      <th>Albuquerque Cycles</th>
      <th>Ann Arbor Speed</th>
      <th>Austin Cruisers</th>
      <th>Cincinnati Speed</th>
      <th>Columbus Race Equipment</th>
      <th>Dallas Cycles</th>
      <th>Denver Bike Shop</th>
      <th>Detroit Cycles</th>
      <th>Indianapolis Velocipedes</th>
      <th>...</th>
      <th>Philadelphia Bike Shop</th>
      <th>Phoenix Bi-peds</th>
      <th>Pittsburgh Mountain Machines</th>
      <th>Portland Bi-peds</th>
      <th>Providence Bi-peds</th>
      <th>San Antonio Bike Shop</th>
      <th>San Francisco Cruisers</th>
      <th>Seattle Race Equipment</th>
      <th>Tampa 29ers</th>
      <th>Wichita Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bad Habit 1</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>27.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bad Habit 2</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>32.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beast of the East 1</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>42.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beast of the East 2</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beast of the East 3</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.0</td>
      <td>23.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
rad = pd.merge(df, merger.drop_duplicates("model"), on="model", how="left")

rad.price.extended = rad.price

rad = rad.drop(["order.date","order.id","order.line","bikeshop.name","quantity","price.extended"],axis=1)

non_cat = list(rad.select_dtypes(exclude=["category","object"]).columns)

cat = list(rad.select_dtypes(include=["category","object"]).columns)

rad[non_cat] = rad[non_cat].fillna(value=0)

```


```python
rad[rad.columns.difference(cat)] = rad[rad.columns.difference(cat)]/rad[rad.columns.difference(cat)].sum()
```

Now we are ready to perform k-means clustering to segment our customer-base. Think of clusters as groups in the customer-base. Prior to starting we will need to choose the number of customer groups, k, that are to be detected. The best way to do this is to think about the customer-base and our hypothesis. We believe that there are most likely to be at least four customer groups because of mountain bike vs road bike and premium vs affordable preferences. We also believe there could be more as some customers may not care about price but may still prefer a specific bike category. However, we’ll limit the clusters to eight as more is likely to overfit the segments. KMeans is really for something that has attributes

Dendogram shows the distance between any two observations in a dataset. The vertical axis determines the distance. The longer the axis, the larger the distance. 


```python
%matplotlib inline
import matplotlib.cm as cm
import seaborn as sn
from sklearn.cluster import KMeans
cmap = sn.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
sn.clustermap(rad.iloc[:,1:-4:].T.head(), cmap=cmap, linewidths=.5)

```




    <seaborn.matrix.ClusterGrid at 0x10ab5c208>




![png](Customer%20Segmentation_files/Customer%20Segmentation_23_1.png)



```python
cluster_range = range(1, 8)
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit(rad.iloc[:,1:-4:].T)
    cluster_errors.append( clusters.inertia_ )
```


```python
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_errors</th>
      <th>num_clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.184222</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.142030</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.121164</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.103948</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.097594</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.089718</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.083149</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

```




    [<matplotlib.lines.Line2D at 0x1a12105c50>]




![png](Customer%20Segmentation_files/Customer%20Segmentation_26_1.png)


Clearly after 4 nothing much happens. You will overfit if you move further away than the elbow.
# I juat compared the last two feature against eachother
# You have to label encode them for below to work
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in rad:
    if rad[col].dtype == 'object' or  rad[col].dtype.name == 'category':
        rad[col] = le.fit_transform(rad[col])

```python
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

cluster_range = range( 2, 7 )

for n_clusters in cluster_range:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(rad.iloc[:,1:-4:]) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict( rad.iloc[:,1:-4:] )

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(rad.iloc[:,1:-4:], cluster_labels)
    print("For n_clusters =", n_clusters,
        "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(rad.iloc[:,1:-4:], cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
      # Aggregate the silhouette scores for samples belonging to
      # cluster i, and sort them
      ith_cluster_silhouette_values = \
          sample_silhouette_values[cluster_labels == i]

      ith_cluster_silhouette_values.sort()

      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      color = cm.spectral(float(i) / n_clusters)
      ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

      # Label the silhouette plots with their cluster numbers at the middle
      ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # Compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(rad.iloc[:, -2], rad.iloc[:, -1], marker='.', s=30, lw=0, alpha=0.7,
              c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
              marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
      ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                "with n_clusters = %d" % n_clusters),
               fontsize=14, fontweight='bold')

    plt.show()
```

    For n_clusters = 2 The average silhouette_score is : 0.19917062862266718



![png](Customer%20Segmentation_files/Customer%20Segmentation_29_1.png)


    For n_clusters = 3 The average silhouette_score is : 0.17962478496274137



![png](Customer%20Segmentation_files/Customer%20Segmentation_29_3.png)


    For n_clusters = 4 The average silhouette_score is : 0.18721328745339205



![png](Customer%20Segmentation_files/Customer%20Segmentation_29_5.png)


    For n_clusters = 5 The average silhouette_score is : 0.19029963290478452



![png](Customer%20Segmentation_files/Customer%20Segmentation_29_7.png)


    For n_clusters = 6 The average silhouette_score is : 0.1665540459580535



![png](Customer%20Segmentation_files/Customer%20Segmentation_29_9.png)


At 4 they are all about the same size, crossing the average line. At 4 number off clusters, the cluster sizes are fairly homogeneous. And only a few observations are assigned to wrong cluster and almost all clusters have observations that are more than the average Silhouette score. 


```python
## Start the clusters here.
```


```python
clusters = KMeans(4)
clusters.fit(rad.iloc[:,1:-4:].T)
#cluster_errors.append( clusters.inertia_ )
centroids = clusters.cluster_centers_
```


```python
clusters = KMeans(4)
clusters.fit(rad.iloc[:,1:-4:])
labels = clusters.predict(rad.iloc[:,1:-4:])
# Centroid values

```


```python
labels
```




    array([0, 0, 0, 0, 0, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
           0, 3, 3, 0, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1,
           1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2,
           0, 0, 0, 0, 0, 3, 3, 3, 3], dtype=int32)




```python
centroids.shape
```




    (4, 97)




```python
## IF I had to traspose, I could've sworns
```


```python
rad.iloc[:,1:-4:].shape
```




    (97, 30)




```python
rad.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>Albuquerque Cycles</th>
      <th>Ann Arbor Speed</th>
      <th>Austin Cruisers</th>
      <th>Cincinnati Speed</th>
      <th>Columbus Race Equipment</th>
      <th>Dallas Cycles</th>
      <th>Denver Bike Shop</th>
      <th>Detroit Cycles</th>
      <th>Indianapolis Velocipedes</th>
      <th>...</th>
      <th>Providence Bi-peds</th>
      <th>San Antonio Bike Shop</th>
      <th>San Francisco Cruisers</th>
      <th>Seattle Race Equipment</th>
      <th>Tampa 29ers</th>
      <th>Wichita Speed</th>
      <th>price</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bad Habit 1</td>
      <td>0.017483</td>
      <td>0.006645</td>
      <td>0.008130</td>
      <td>0.005115</td>
      <td>0.010152</td>
      <td>0.012821</td>
      <td>0.011734</td>
      <td>0.009921</td>
      <td>0.006270</td>
      <td>...</td>
      <td>0.009225</td>
      <td>0.021505</td>
      <td>0.002674</td>
      <td>0.015625</td>
      <td>0.019417</td>
      <td>0.005917</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bad Habit 2</td>
      <td>0.006993</td>
      <td>0.009967</td>
      <td>0.004065</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017094</td>
      <td>0.013907</td>
      <td>0.015873</td>
      <td>0.003135</td>
      <td>...</td>
      <td>0.023985</td>
      <td>0.000000</td>
      <td>0.002674</td>
      <td>0.007812</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beast of the East 1</td>
      <td>0.010490</td>
      <td>0.014950</td>
      <td>0.008130</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004274</td>
      <td>0.018253</td>
      <td>0.011905</td>
      <td>0.009404</td>
      <td>...</td>
      <td>0.009225</td>
      <td>0.005376</td>
      <td>0.000000</td>
      <td>0.015625</td>
      <td>0.009709</td>
      <td>0.000000</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beast of the East 2</td>
      <td>0.010490</td>
      <td>0.009967</td>
      <td>0.008130</td>
      <td>0.000000</td>
      <td>0.005076</td>
      <td>0.004274</td>
      <td>0.015211</td>
      <td>0.005952</td>
      <td>0.009404</td>
      <td>...</td>
      <td>0.014760</td>
      <td>0.010753</td>
      <td>0.002674</td>
      <td>0.023438</td>
      <td>0.029126</td>
      <td>0.001972</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beast of the East 3</td>
      <td>0.003497</td>
      <td>0.003322</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002538</td>
      <td>0.004274</td>
      <td>0.016949</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.007380</td>
      <td>0.032258</td>
      <td>0.000000</td>
      <td>0.007812</td>
      <td>0.009709</td>
      <td>0.000000</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
for i, c in enumerate(centroids):
    rad["Cluster "+str(i)] = list(c)
```


```python
rad_final = rad.drop(list(rad.iloc[:,1:].iloc[:,:-8].columns),axis=1)
```


```python
rad_final.sort_values("Cluster 0").head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>price</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>Cluster 0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>Synapse Hi-Mod Disc Black Inc.</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>0.003503</td>
      <td>0.009168</td>
      <td>0.019755</td>
      <td>0.007196</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Supersix Evo Black Inc.</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>0.003620</td>
      <td>0.008226</td>
      <td>0.019639</td>
      <td>0.015364</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Slice Hi-Mod Black Inc.</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Triathalon</td>
      <td>Carbon</td>
      <td>0.003827</td>
      <td>0.016406</td>
      <td>0.023494</td>
      <td>0.013543</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Supersix Evo Hi-Mod Dura Ace 1</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>0.003856</td>
      <td>0.005667</td>
      <td>0.023053</td>
      <td>0.006798</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Synapse Hi-Mod Dura Ace</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>0.004202</td>
      <td>0.013474</td>
      <td>0.021672</td>
      <td>0.008991</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Synapse Carbon Disc Ultegra D12</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>0.004295</td>
      <td>0.012610</td>
      <td>0.019516</td>
      <td>0.012696</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Jekyll Carbon 3</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>0.004389</td>
      <td>0.013288</td>
      <td>0.011734</td>
      <td>0.000524</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Supersix Evo Hi-Mod Team</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>0.004652</td>
      <td>0.008754</td>
      <td>0.016159</td>
      <td>0.013339</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CAAD12 Black Inc</td>
      <td>(2700.0, 12790.0]</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>0.004709</td>
      <td>0.004142</td>
      <td>0.019471</td>
      <td>0.011152</td>
    </tr>
    <tr>
      <th>26</th>
      <td>F-Si Hi-Mod 1</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>0.004748</td>
      <td>0.013929</td>
      <td>0.010030</td>
      <td>0.000676</td>
    </tr>
  </tbody>
</table>
</div>




```python
rad_final = rad_final.rename(columns={"Cluster 0":"Low End Road Bike Customer"})
```


```python
rad_final.sort_values("Cluster 1").head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>price</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>Low End Road Bike Customer</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>Slice Ultegra</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Triathalon</td>
      <td>Carbon</td>
      <td>0.018821</td>
      <td>0.000000</td>
      <td>0.013370</td>
      <td>0.022077</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Syapse Carbon Tiagra</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>0.010617</td>
      <td>0.000000</td>
      <td>0.010837</td>
      <td>0.018366</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Slice 105</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Triathalon</td>
      <td>Carbon</td>
      <td>0.013984</td>
      <td>0.000000</td>
      <td>0.011002</td>
      <td>0.017242</td>
    </tr>
    <tr>
      <th>60</th>
      <td>SuperX Rival CX1</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Cyclocross</td>
      <td>Carbon</td>
      <td>0.014142</td>
      <td>0.000000</td>
      <td>0.010074</td>
      <td>0.013548</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Synapse Carbon Ultegra 4</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>0.013919</td>
      <td>0.000000</td>
      <td>0.011632</td>
      <td>0.020372</td>
    </tr>
    <tr>
      <th>58</th>
      <td>SuperX 105</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Cyclocross</td>
      <td>Carbon</td>
      <td>0.012159</td>
      <td>0.000000</td>
      <td>0.009596</td>
      <td>0.014853</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CAAD8 Sora</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>0.015626</td>
      <td>0.000000</td>
      <td>0.013606</td>
      <td>0.017489</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Supersix Evo Tiagra</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>0.012011</td>
      <td>0.000264</td>
      <td>0.013232</td>
      <td>0.018053</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CAAD12 Disc 105</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>0.016751</td>
      <td>0.000264</td>
      <td>0.014013</td>
      <td>0.015795</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Synapse Disc Tiagra</td>
      <td>(414.999, 2700.0]</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Aluminum</td>
      <td>0.013016</td>
      <td>0.000264</td>
      <td>0.008813</td>
      <td>0.024608</td>
    </tr>
  </tbody>
</table>
</div>




```python
rad_final = rad_final.rename(columns={"Cluster 1":"High End Road Bike Customer"})
```


```python
rad_final.sort_values("Cluster 2").head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>price</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>Low End Road Bike Customer</th>
      <th>High End Road Bike Customer</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>F-Si 1</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Aluminum</td>
      <td>0.012330</td>
      <td>0.017409</td>
      <td>-1.734723e-18</td>
      <td>0.007079</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Habit 4</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>0.015225</td>
      <td>0.011251</td>
      <td>0.000000e+00</td>
      <td>0.004316</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Fat CAAD2</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Fat Bike</td>
      <td>Aluminum</td>
      <td>0.012995</td>
      <td>0.007621</td>
      <td>1.734723e-18</td>
      <td>0.006693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beast of the East 1</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>0.012339</td>
      <td>0.012125</td>
      <td>2.670940e-04</td>
      <td>0.012970</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Catalyst 3</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>0.018260</td>
      <td>0.003200</td>
      <td>2.670940e-04</td>
      <td>0.007307</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Catalyst 1</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>0.014337</td>
      <td>0.006794</td>
      <td>3.287311e-04</td>
      <td>0.008686</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Habit 6</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>0.014141</td>
      <td>0.004235</td>
      <td>4.230118e-04</td>
      <td>0.010441</td>
    </tr>
    <tr>
      <th>25</th>
      <td>F-Si Carbon 4</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>0.018444</td>
      <td>0.008732</td>
      <td>4.230118e-04</td>
      <td>0.010026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bad Habit 2</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>0.012405</td>
      <td>0.004576</td>
      <td>4.456328e-04</td>
      <td>0.008157</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Trail 1</td>
      <td>(414.999, 2700.0]</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>0.015381</td>
      <td>0.005553</td>
      <td>6.901059e-04</td>
      <td>0.013061</td>
    </tr>
  </tbody>
</table>
</div>




```python
rad_final = rad_final.rename(columns={"Cluster 2":"Aluminum Mountain Bike Customers"})
```


```python
rad_final.sort_values("Cluster 3").head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>price</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>Low End Road Bike Customer</th>
      <th>High End Road Bike Customer</th>
      <th>Aluminum Mountain Bike Customers</th>
      <th>Cluster 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>Habit Hi-Mod Black Inc.</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Carbon</td>
      <td>0.007331</td>
      <td>0.011520</td>
      <td>0.008958</td>
      <td>0.000316</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Jekyll Carbon 1</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>0.008027</td>
      <td>0.018505</td>
      <td>0.008085</td>
      <td>0.000487</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Jekyll Carbon 3</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>0.004389</td>
      <td>0.013288</td>
      <td>0.011734</td>
      <td>0.000524</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Habit Carbon 2</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Carbon</td>
      <td>0.005371</td>
      <td>0.023375</td>
      <td>0.007472</td>
      <td>0.000528</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Scalpel-Si Race</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>0.011713</td>
      <td>0.009638</td>
      <td>0.012867</td>
      <td>0.000569</td>
    </tr>
    <tr>
      <th>26</th>
      <td>F-Si Hi-Mod 1</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>0.004748</td>
      <td>0.013929</td>
      <td>0.010030</td>
      <td>0.000676</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Jekyll Carbon 2</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>0.006515</td>
      <td>0.021079</td>
      <td>0.010311</td>
      <td>0.000744</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Habit Carbon 3</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Carbon</td>
      <td>0.007421</td>
      <td>0.008960</td>
      <td>0.011745</td>
      <td>0.000785</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Scalpel-Si Carbon 3</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>0.005525</td>
      <td>0.034269</td>
      <td>0.012994</td>
      <td>0.000800</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Scalpel-Si Hi-Mod 1</td>
      <td>(2700.0, 12790.0]</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>0.006579</td>
      <td>0.018085</td>
      <td>0.013578</td>
      <td>0.000862</td>
    </tr>
  </tbody>
</table>
</div>




```python
rad_final = rad_final.rename(columns={"Cluster 3":"High End Carbon Mountain Bike Customers"})
```

If you review your results and some of the clusters happened to be similar, it might be necessary to drop a few clusters and rerun the algorithm. In our case tha is not necessary, as generally good separations have been done. It is good to remember that the customer segmentation process can be performed with various clustering algorithms. In this post, we focused on k-means clustering 

PCA is nothing more than an algorithm that takes numeric data in x, y, z coordinates and changes the coordinates to x’, y’, and z’ that maximize the linear variance.

How does this help in customer segmentation / community detection? Unlike k-means, PCA is not a direct solution. What PCA helps with is visualizing the essence of a data set. Because PCA selects PC’s based on the maximum linear variance, we can use the first few PC’s to describe a vast majority of the data set without needing to compare and contrast every single feature. By using PC1 and PC2, we can then visualize in 2D and inspect for clusters. We can also combine the results with the k-means groups to see what k-means detected as compared to the clusters in the PCA visualization.

If you want to scale the data, a scaler function is fine, if you want to scale and center the data then standardisation is the best. In practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation. This is called standardisation. 


```python
from sklearn.pipeline import make_pipeline

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#pca2 = PCA(n_components=2)

pca2_results = make_pipeline(StandardScaler(),PCA(n_components=2)).fit_transform(rad.iloc[:,1:-(4+len(centroids))])

for i in range(pca2_results.shape[1]):
    rad.iloc[:,1:-(4+len(centroids)):]["pca_"+str(i)] = pca2_results[:,i]

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots(figsize=(20,15))
points = ax.scatter(pca2_results[:,0], pca2_results[:,1],c=labels,  s=50, cmap=cmap)
#c=df_2.TARGET,
f.colorbar(points)
plt.show()
### Each dot is a cycle shop
```


![png](Customer%20Segmentation_files/Customer%20Segmentation_51_0.png)


PCA can be a valuable cross-check to k-means for customer segmentation. While k-means got us close to the true customer segments, visually evaluating the groups using PCA helped identify a different customer segment, one that the [Math Processing Error] k-means solution did not pick up.

For customer segmentation, we can utilize network visualization to understand both the network communities and the strength of the relationships. Before we jump into network visualization

The first step to network visualization is to get the data organized into a cosine similarity matrix. A similarity matrix is a way of numerically representing the similarity between multiple variables similar to a correlation matrix. We’ll use Cosine Similarity to measure the relationship, which measures how similar the direction of a vector is to another vector. If that seems complicated, just think of a customer cosine similarity as a number that reflects how closely the direction of buying habits are related. Numbers will range from zero to one with numbers closer to one indicating very similar buying habits and numbers closer to zero indicating dissimilar buying habits.


```python
from sklearn.metrics.pairwise import cosine_similarity
```


```python
import numpy as np
```


```python
cosine_similarity(rad.iloc[:,1:-(4+len(centroids))].T).shape
```




    (30, 30)




```python
cos_mat = pd.DataFrame(cosine_similarity(rad.iloc[:,1:-(4+len(centroids))].T), index=list(rad.iloc[:,1:-(4+len(centroids))].columns), columns=list(rad.iloc[:,1:-(4+len(centroids))].columns))
```


```python
## Make diagonal zero
cos_mat.values[[np.arange(len(cos_mat))]*2] = 0
```


```python
def front(self, n):
    return self.iloc[:, :n]

pd.DataFrame.front = front
```


```python
cos_mat.head(5).front(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Albuquerque Cycles</th>
      <th>Ann Arbor Speed</th>
      <th>Austin Cruisers</th>
      <th>Cincinnati Speed</th>
      <th>Columbus Race Equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albuquerque Cycles</th>
      <td>0.000000</td>
      <td>0.619604</td>
      <td>0.594977</td>
      <td>0.544172</td>
      <td>0.582015</td>
    </tr>
    <tr>
      <th>Ann Arbor Speed</th>
      <td>0.619604</td>
      <td>0.000000</td>
      <td>0.743195</td>
      <td>0.719272</td>
      <td>0.659216</td>
    </tr>
    <tr>
      <th>Austin Cruisers</th>
      <td>0.594977</td>
      <td>0.743195</td>
      <td>0.000000</td>
      <td>0.594016</td>
      <td>0.566944</td>
    </tr>
    <tr>
      <th>Cincinnati Speed</th>
      <td>0.544172</td>
      <td>0.719272</td>
      <td>0.594016</td>
      <td>0.000000</td>
      <td>0.795889</td>
    </tr>
    <tr>
      <th>Columbus Race Equipment</th>
      <td>0.582015</td>
      <td>0.659216</td>
      <td>0.566944</td>
      <td>0.795889</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



It’s a good idea to prune the tree before we move to graphing. The network graphs can become quite messy if we do not limit the number of edges. We do this by reviewing the cosine similarity matrix and selecting an edgeLimit, a number below which the cosine similarities will be replaced with zero. This keeps the highest ranking relationships while reducing the noise. We select 0.70 as the limit, but typically this is a trial and error process. If the limit is too high, the network graph will not show enough detail. 


```python
edgeLimit = 0.7
cos_mat = cos_mat.applymap(lambda x: 0 if x <edgeLimit else x)
```


```python
cos_mat.head(5).front(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Albuquerque Cycles</th>
      <th>Ann Arbor Speed</th>
      <th>Austin Cruisers</th>
      <th>Cincinnati Speed</th>
      <th>Columbus Race Equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albuquerque Cycles</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ann Arbor Speed</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.743195</td>
      <td>0.719272</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Austin Cruisers</th>
      <td>0.0</td>
      <td>0.743195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Cincinnati Speed</th>
      <td>0.0</td>
      <td>0.719272</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.795889</td>
    </tr>
    <tr>
      <th>Columbus Race Equipment</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.795889</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import igraph

```


```python
from scipy.cluster.hierarchy import dendrogram, linkage
```


```python
## I think the diagonal creates the one drpop
Z = linkage(cos_mat, 'ward')
```

    /Users/dereksnow/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel/__main__.py:2: ClusterWarning: scipy.cluster: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
      from ipykernel import kernelapp as app



```python
cos_mat.drop_duplicates().shape
```




    (30, 30)




```python
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(cos_mat))
```


```python
#list(cos_mat.as_matrix())
```


```python
cos_mat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Albuquerque Cycles</th>
      <th>Ann Arbor Speed</th>
      <th>Austin Cruisers</th>
      <th>Cincinnati Speed</th>
      <th>Columbus Race Equipment</th>
      <th>Dallas Cycles</th>
      <th>Denver Bike Shop</th>
      <th>Detroit Cycles</th>
      <th>Indianapolis Velocipedes</th>
      <th>Ithaca Mountain Climbers</th>
      <th>...</th>
      <th>Philadelphia Bike Shop</th>
      <th>Phoenix Bi-peds</th>
      <th>Pittsburgh Mountain Machines</th>
      <th>Portland Bi-peds</th>
      <th>Providence Bi-peds</th>
      <th>San Antonio Bike Shop</th>
      <th>San Francisco Cruisers</th>
      <th>Seattle Race Equipment</th>
      <th>Tampa 29ers</th>
      <th>Wichita Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albuquerque Cycles</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.700233</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.730533</td>
      <td>0.000000</td>
      <td>0.707184</td>
      <td>0.721538</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ann Arbor Speed</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.743195</td>
      <td>0.719272</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.738650</td>
      <td>0.756429</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.773410</td>
      <td>0.000000</td>
      <td>0.721959</td>
      <td>0.782233</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.704031</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Austin Cruisers</th>
      <td>0.000000</td>
      <td>0.743195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.752929</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.717374</td>
      <td>0.771772</td>
      <td>0.000000</td>
      <td>0.746299</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Cincinnati Speed</th>
      <td>0.000000</td>
      <td>0.719272</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.795889</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.829649</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.807522</td>
    </tr>
    <tr>
      <th>Columbus Race Equipment</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.795889</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.704296</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.778459</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.748018</td>
    </tr>
    <tr>
      <th>Dallas Cycles</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.743126</td>
      <td>0.749603</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.768081</td>
      <td>0.000000</td>
      <td>0.754692</td>
      <td>0.756797</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Denver Bike Shop</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.743126</td>
      <td>0.000000</td>
      <td>0.784315</td>
      <td>0.000000</td>
      <td>0.739835</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.873259</td>
      <td>0.000000</td>
      <td>0.855332</td>
      <td>0.795989</td>
      <td>0.728577</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Detroit Cycles</th>
      <td>0.700233</td>
      <td>0.738650</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.704296</td>
      <td>0.749603</td>
      <td>0.784315</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.702683</td>
      <td>0.833591</td>
      <td>0.000000</td>
      <td>0.836725</td>
      <td>0.793167</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Indianapolis Velocipedes</th>
      <td>0.000000</td>
      <td>0.756429</td>
      <td>0.752929</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.714849</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ithaca Mountain Climbers</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.739835</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.814699</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.766426</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Kansas City 29ers</th>
      <td>0.709368</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.746897</td>
      <td>0.957257</td>
      <td>0.788795</td>
      <td>0.000000</td>
      <td>0.738281</td>
      <td>...</td>
      <td>0.704455</td>
      <td>0.885133</td>
      <td>0.000000</td>
      <td>0.863437</td>
      <td>0.826441</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Las Vegas Cycles</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.819100</td>
      <td>0.793809</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.798941</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.857356</td>
    </tr>
    <tr>
      <th>Los Angeles Cycles</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.765069</td>
      <td>0.701997</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.813920</td>
      <td>0.000000</td>
      <td>0.803344</td>
      <td>0.729771</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Louisville Race Equipment</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.858299</td>
      <td>0.780713</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.753219</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.819657</td>
    </tr>
    <tr>
      <th>Miami Race Equipment</th>
      <td>0.000000</td>
      <td>0.887012</td>
      <td>0.777433</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.766868</td>
      <td>0.746176</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.715803</td>
      <td>0.843452</td>
      <td>0.000000</td>
      <td>0.793937</td>
      <td>0.788819</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.747068</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Minneapolis Bike Shop</th>
      <td>0.700426</td>
      <td>0.744016</td>
      <td>0.736987</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.780322</td>
      <td>0.802102</td>
      <td>0.780967</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.708330</td>
      <td>0.895092</td>
      <td>0.000000</td>
      <td>0.852946</td>
      <td>0.833356</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Nashville Cruisers</th>
      <td>0.000000</td>
      <td>0.812436</td>
      <td>0.753784</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.737619</td>
      <td>0.700912</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.798585</td>
      <td>0.000000</td>
      <td>0.773644</td>
      <td>0.733613</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.726407</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>New Orleans Velocipedes</th>
      <td>0.000000</td>
      <td>0.850807</td>
      <td>0.809239</td>
      <td>0.721739</td>
      <td>0.000000</td>
      <td>0.705624</td>
      <td>0.000000</td>
      <td>0.783405</td>
      <td>0.780344</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.836748</td>
      <td>0.000000</td>
      <td>0.808124</td>
      <td>0.773396</td>
      <td>0.000000</td>
      <td>0.703193</td>
      <td>0.811988</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>New York Cycles</th>
      <td>0.000000</td>
      <td>0.731991</td>
      <td>0.708396</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.787456</td>
      <td>0.753576</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.701388</td>
      <td>0.849327</td>
      <td>0.000000</td>
      <td>0.816018</td>
      <td>0.771974</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.722622</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Oklahoma City Race Equipment</th>
      <td>0.000000</td>
      <td>0.875217</td>
      <td>0.823806</td>
      <td>0.714466</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.782434</td>
      <td>0.771407</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.705122</td>
      <td>0.857586</td>
      <td>0.000000</td>
      <td>0.828342</td>
      <td>0.798605</td>
      <td>0.715004</td>
      <td>0.000000</td>
      <td>0.805118</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Philadelphia Bike Shop</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.717374</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.702683</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.734866</td>
      <td>0.000000</td>
      <td>0.764986</td>
      <td>0.000000</td>
      <td>0.720375</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Phoenix Bi-peds</th>
      <td>0.730533</td>
      <td>0.773410</td>
      <td>0.771772</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.768081</td>
      <td>0.873259</td>
      <td>0.833591</td>
      <td>0.714849</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.734866</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.913740</td>
      <td>0.879047</td>
      <td>0.740801</td>
      <td>0.000000</td>
      <td>0.722206</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Pittsburgh Mountain Machines</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.814699</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.716919</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Portland Bi-peds</th>
      <td>0.707184</td>
      <td>0.721959</td>
      <td>0.746299</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.754692</td>
      <td>0.855332</td>
      <td>0.836725</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.764986</td>
      <td>0.913740</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.815651</td>
      <td>0.760300</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Providence Bi-peds</th>
      <td>0.721538</td>
      <td>0.782233</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.756797</td>
      <td>0.795989</td>
      <td>0.793167</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.879047</td>
      <td>0.000000</td>
      <td>0.815651</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>San Antonio Bike Shop</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.728577</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.720375</td>
      <td>0.740801</td>
      <td>0.000000</td>
      <td>0.760300</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>San Francisco Cruisers</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.829649</td>
      <td>0.778459</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.779465</td>
    </tr>
    <tr>
      <th>Seattle Race Equipment</th>
      <td>0.000000</td>
      <td>0.704031</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.722206</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Tampa 29ers</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.766426</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.716919</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Wichita Speed</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.807522</td>
      <td>0.748018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.779465</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 30 columns</p>
</div>




```python
Z.shape
```




    (29, 4)




```python
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    labels = list(cos_mat.columns),
)
plt.show()

## You can create a reasonable argument for 4 - 6 clusters. 
```


![png](Customer%20Segmentation_files/Customer%20Segmentation_73_0.png)



```python
cos_mat["Detroit Cycles"].sort_values(ascending=False)
```




    Portland Bi-peds                0.836725
    Phoenix Bi-peds                 0.833591
    Providence Bi-peds              0.793167
    Kansas City 29ers               0.788795
    Denver Bike Shop                0.784315
    New Orleans Velocipedes         0.783405
    Oklahoma City Race Equipment    0.782434
    Minneapolis Bike Shop           0.780967
    Miami Race Equipment            0.766868
    New York Cycles                 0.753576
    Dallas Cycles                   0.749603
    Ann Arbor Speed                 0.738650
    Nashville Cruisers              0.737619
    Columbus Race Equipment         0.704296
    Philadelphia Bike Shop          0.702683
    Los Angeles Cycles              0.701997
    Albuquerque Cycles              0.700233
    Louisville Race Equipment       0.000000
    Las Vegas Cycles                0.000000
    Tampa 29ers                     0.000000
    Ithaca Mountain Climbers        0.000000
    Indianapolis Velocipedes        0.000000
    Detroit Cycles                  0.000000
    Pittsburgh Mountain Machines    0.000000
    San Antonio Bike Shop           0.000000
    San Francisco Cruisers          0.000000
    Cincinnati Speed                0.000000
    Austin Cruisers                 0.000000
    Seattle Race Equipment          0.000000
    Wichita Speed                   0.000000
    Name: Detroit Cycles, dtype: float64




1. horizontal lines are cluster merges
2. vertical lines tell you which clusters/labels were part of merge forming that new cluster
3. heights of the horizontal lines tell you about the distance that needed to be "bridged" to form the new cluster


In case you're wondering about where the colors come from, you might want to have a look at the color_threshold argument of dendrogram(), which as not specified automagically picked a distance cut-off value of 70 % of the final merge and then colored the first clusters below that in individual colors.


```python
#### Supposedly, you can do some network analysis here.  Of course, you are not sure how. 

cos_igraph = igraph.Graph.Adjacency(list(cos_mat.as_matrix()),mode = 'undirected')

cos_bet = igraph.Graph.edge_betweenness(cos_igraph)

cos_bet

plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(cos_igraph)
plt.show()
```


```python
## Possibly use this for future network:
https://python-graph-gallery.com/327-network-from-correlation-matrix/
    
    
```
