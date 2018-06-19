# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:39:48 2018

@author: Hrithik
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
china=load_sample_image("flower.jpg")
ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china)
print("(Height of image,Width of Image,RGB value[0-255]) =",china.shape)

data=china/255
data=data.reshape(427*640,3)
print("Viewing this set of pixels as a clous of points in a 3D color space")
print("Rescaling the colos such that they lie bw 0 and 1")
print("Visualising these pixels in a color space using a subset of 10,000 pixels for etaPer")

def plot_pixels(data,title,colors=None,N=10000):
    if colors is None:
        colors=data
    rng=np.random.RandomState(0)
    i=rng.permutation(data.shape[0])[:N]
    colors=colors[i]
    R,G,B=data[i].T
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    ax[0].scatter(R,G,color=colors,marker='.')
    ax[0].set(xlabel='Red',ylabel='Green',xlim=(0,1),ylim=(0,1))
    ax[1].scatter(R,B,color=colors,marker='.')
    ax[1].set(xlabel='Red',ylabel='Blue',xlim=(0,1),ylim=(0,1))
    
    fig.suptitle(title,size=20)
plot_pixels(data,title='input space is 16 million possible colors ')

import warnings
warnings.simplefilter('ignore')
from sklearn.cluster import MiniBatchKMeans
kmeans=MiniBatchKMeans(16)
kmeans.fit(data)
new_colors=kmeans.cluster_centers_[kmeans.predict(data)]
plot_pixels(data,colors=new_colors,title='Reduced color space to 16 colors')

china_recolored=new_colors.reshape(china.shape)
fig,ax=plt.subplots(1,2,figsize=(16,6),subplot_kw=dict(xticks=[],yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('original image with 16 million colors',size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('Recolored image with 16 ',size=16)


ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china_recolored)
































