# Clustering(Unsupervised)
Coded Examples of Different types of Clustering Techniques...

# K-means Clustering-
Digit Recognition...Well it do seems to wear off upon convoluted encounters...Hence we got..

# Spectral Clustering-open KMeansClustering.py for side by side comparision
Plays pretty good with complex geometry but still, what if we have like- Million DataPoints?

# Mini-Batch k-means-
baddest off em' all...Color Quantization..Set k as per ur need..!

# K-means elbow method-
Well bruh..do u really know,how many clusters u need?..Go for the elbow! 

# NOTE !!!
paste this little shite in DigitRecognitionKMeans.py for generating the Heatmap..(A cooler terminology for Confusion Matrix)

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(digits.target,labels)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar='False',
            xticklabels=digits.target_names,yticklabels=digits.target_names)
plt.xlabel("True Label")
plt.ylabel("False Label")
