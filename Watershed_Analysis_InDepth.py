#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports

import skimage
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

import h5py
import cv2
import numpy as np
import sys
from PIL import Image
import skimage.external.tifffile as tfl
import scipy
import skimage.segmentation
# import skimage.io
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label, generate_binary_structure
from mpl_toolkits.mplot3d import Axes3D

# In[2]:


# from https://github.com/divelab/crnn/blob/master/train_predict.py

def watershed_1(prob_b):
	from scipy import ndimage as ndi
	from skimage.morphology import watershed
	from skimage.feature import peak_local_max
	binary_boundary = np.round(prob_b)
	distance = ndi.distance_transform_edt(binary_boundary)
	local_maxi = peak_local_max(distance, indices=False,
                            labels=binary_boundary)
	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=binary_boundary)
	return labels


# In[3]:


def watershed_2(prob):
    mask_prob = prob
    mask_prob[mask_prob >= 0.7] = 1 
    mask_prob[mask_prob < 0.7] = 0
    
    distance_2d = ndi.distance_transform_edt(prob)
    local_maxi_2d = peak_local_max(distance_2d, indices=False,footprint=np.ones((3, 3)),
                            labels=prob,min_distance=2)
    markers_2d = ndi.label(local_maxi_2d)[0]
    results_2d = watershed(-distance_2d, markers_2d, mask=mask_prob)
    return results_2d


# In[4]:


def watershed_3(prob):
    mask_prob = prob
    mask_prob[mask_prob >= 0.7] = 1 
    mask_prob[mask_prob < 0.7] = 0
    
    sd = 1
    window = 6
    t = (((window - 1)/2)-0.5)/sd
    filtered_img = gaussian_filter(prob, sigma=sd, truncate=t)
    distance = ndi.distance_transform_edt(filtered_img)
    ws_out = skimage.segmentation.watershed(-distance,markers=filtered_img,mask=mask_prob)
    return ws_out


# In[5]:


def watershed_4(prob):
    mask_prob = prob
    mask_prob[mask_prob >= 0.7] = 1 
    mask_prob[mask_prob < 0.7] = 0
    
    sd = 1
    window = 6
    t = (((window - 1)/2)-0.5)/sd
    filtered_img = gaussian_filter(prob, sigma=sd, truncate=t)
    minima = skimage.morphology.extrema.h_minima(filtered_img, 0.14)
    
    distance = ndi.distance_transform_edt(filtered_img)
    ws_out = skimage.segmentation.watershed(-distance,markers=minima,mask=mask_prob)
    return ws_out


# In[6]:


def watershed_5(prob):
    mask_prob = prob
    mask_prob[mask_prob >= 0.7] = 1 
    mask_prob[mask_prob < 0.7] = 0
    labeled_array, num_features = label(mask_prob)
    return labeled_array


# In[7]:


def watershed_6(prob):
    labeled_img = skimage.morphology.label(prob, neighbors=4)
    return labeled_img


# In[8]:


FILE_PATH = '/home/manish/TAMU_SUMMER_2019/deepem3d/'
FILE_LABEL_PATH = '/home/manish/TAMU_SUMMER_2019/data_files/'
PROB_FILE = 'prob_snemi.h5'
LABEL_FILE = 'train_labelEM.h5'



# In[9]:


prob = h5py.File(FILE_PATH+PROB_FILE, 'r')
prob = np.array(prob['data'])

out_label = h5py.File(FILE_LABEL_PATH + LABEL_FILE, 'r')
out_label = out_label['label']


# In[10]:


# single instances
prob_single = prob[:,:,0]

label_single = out_label[:,:,0]


# In[11]:


def multiplot(x,y,img_array):
    fig, axes = plt.subplots(x,y, sharex=True,sharey=True,figsize=(20,10))
    for i in range(x*y):
        axes[i].imshow(img_array[i],extent=[0,1023,0,1023])
    axes[0].set_title('Segmentation')
    axes[1].set_title('Ground truth')
    plt.show()
    return


# In[12]:


ws1 = watershed_1(prob_single)


# In[13]:


ws2 = watershed_2(prob_single)


# In[14]:


ws3 = watershed_3(prob_single)


# In[15]:


ws4 = watershed_4(prob_single)


# In[16]:


ws5 = watershed_5(prob_single)


# In[17]:


ws6 = watershed_6(prob_single)


# In[18]:


#ws_arr = [ws1,label_single]
ws_arr = [ws6,label_single]


# In[19]:


# The following plots show the result of the different watershed algo. Below this we have 
# the number of features or neurons


# In[20]:


#multiplot(1,2,ws_arr)
def ImageSliceViews(prob,out_label):
    ws_3d = watershed_1(prob)

    class IndexTracker(object):
        def __init__(self, ax, X,Y):
            self.ax = ax[0]
            self.ay = ax[1]

            self.X = X
            self.Y = Y
            rows, cols, self.slices = X.shape
            self.ind = self.slices//2

            self.imx = ax[0].imshow(self.X[:, :, self.ind],extent=[0,1023,0,1023])
            self.imy = ax[1].imshow(self.Y[:, :, self.ind],extent=[0,1023,0,1023])
            self.ax.set_title('Segmentation')
            self.ay.set_title('Ground truth')

           #self.axes[i].imshow(img_array[i],extent=[0,1023,0,1023])
            self.update()

        def onscroll(self, event):
            print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            self.imx.set_data(self.X[:, :, self.ind])
            self.imy.set_data(self.Y[:, :, self.ind])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.ay.set_ylabel('slice %s' % self.ind)
            self.imx.axes.figure.canvas.draw()
            self.imy.axes.figure.canvas.draw()


    fig, ax = plt.subplots(1,2, sharex=True,sharey=True)


    tracker = IndexTracker(ax, ws_3d,out_label)


    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

ImageSliceViews(prob,out_label)
# In[21]:


# Number of features in each watershed
num_features_1= np.unique(ws1).size
print("nf1= "+ str(num_features_1))
# The ground truth label along with the number of features:

plt.imshow(label_single)

num_features_label = np.unique(label_single).size
print("nf output= "+str(num_features_label))





