import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
X,y_true=make_blobs(n_samples=300,centers=4,random_state=0,cluster_std=0.89)
plt.scatter(X[:,0],X[:,1],s=50)
plt.show()

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)
centers=kmeans.cluster_centers_
plt.scatter(X[:,0],X[:,1],c=y_kmeans,cmap='viridis',s=50)
plt.scatter(centers[:,0],centers[:,1],c='black',s=200)
plt.show()


from sklearn.datasets import make_moons
a,b=make_moons(200,noise=0.05,random_state=0)
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
labels=model.fit_predict(a)
plt.scatter(a[:,0],a[:,1],c=labels,cmap='viridis',s=50)
plt.show()



from sklearn.datasets import make_moons
c,d=make_moons(200,noise=0.05,random_state=0)
from sklearn.cluster import SpectralClustering
model=SpectralClustering(n_clusters=2,affinity='nearest_neighbors',assign_labels='kmeans')
labels=model.fit_predict(c)
plt.scatter(c[:,0],c[:,1],c=labels,cmap='viridis',s=50)
plt.show()
print("Spectral Clustering did a good job than Simple K-means..Well u do see the moons.Right?")






