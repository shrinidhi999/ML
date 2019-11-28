import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('.\Datasets\Mall_Customers.csv')
X = df.iloc[:,[3,4]].values

##train and test split

from sklearn.cluster import KMeans
wcss =[]

for i in range(1,11):
    km = KMeans(n_clusters=i,init='k-means++',random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

x_grid = np.arange(1,11,1).astype(int)
plt.plot(x_grid, wcss)
plt.show()

km = KMeans(n_clusters=4,random_state=42)

y_kmeans = km.fit_predict(X)

# print(X[0,0])

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()