# =======================   KMEans on numbers   =================================
from sklearn.datasets import load_digits
digits=load_digits()
kmean=KMeans(n_clusters=10,random_state=0)
clusters=kmean.fit_predict(digits.data)
#Finding how the cluster centers looks like
fig,ax=plt.subplots(2,5,figsize=(8,3))
centersL=kmean.cluster_centers_.reshape(10,8,8)
for axi,center in zip(ax.flat,centersL):
    axi.set(xticks=[],yticks=[])
    axi.imshow(center,interpolation='nearest',cmap=plt.cm.binary)
print("Even without the labels kmeans was able to identify clusters centers as recognizable digits.")
#-----------------------=-Finding accuracy for KMEans========-=-=--------------
from scipy.stats import mode
labels=np.zeros_like(clusters)
for i in range (10):
    mask=(clusters==i)
    labels[mask]=mode(digits.target[mask])[0]
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(digits.target,labels)
print("Even with k means we get an accuracy of",accuracy)

# ===================Pre process data before kmeans====================================

from sklearn.manifold import TSNE
print("Performing T distributed stochastic neighbor embedding(t-SNE)")
tsne=TSNE(n_components=2,init='random',random_state=0)
its_proj=tsne.fit_transform(digits.data)
kmean=KMeans(n_clusters=10,random_state=0)
clusters=kmean.fit_predict(its_proj)
#Finding how the cluster centers looks like
fig,ax=plt.subplots(2,5,figsize=(8,3))
for axi,center in zip(ax.flat,centersL):
    axi.set(xticks=[],yticks=[])
    axi.imshow(center,interpolation='nearest',cmap=plt.cm.binary)
from scipy.stats import mode
labels=np.zeros_like(clusters)
for i in range (10):
    mask=(clusters==i)
    labels[mask]=mode(digits.target[mask])[0]
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(digits.target,labels)
print("After preproceesing data under t-SNE we get ",accuracy)





# =============================================================================
# =============================================================================