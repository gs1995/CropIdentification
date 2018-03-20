
# coding: utf-8

# In[25]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import os
import scipy

from matplotlib import pyplot as plt
from matplotlib import colors
#from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, felzenszwalb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

RASTER_DATA_FILE = "data/image/2298119ene2016recorteTT.tif"
TRAIN_DATA_PATH = "data/train/"
TEST_DATA_PATH = "data/test/"


# In[28]:


from pylab import *
verification_pixels=load("test.npy")
print(verification_pixels.shape)
for_verification = np.nonzero(verification_pixels)


# In[29]:


verification_labels = verification_pixels[for_verification]
predicted_labels = clf[for_verification]


# In[21]:


files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.shp')]
classes_labels = [f.split('.')[0] for f in files]
#print(classes_labels)
segments=load("segments.npy")
clf = np.copy(segments)
for_verification = np.nonzero(verification_pixels)
verification_labels = verification_pixels[for_verification]
predicted_labels = clf[for_verification]
cm = metrics.confusion_matrix(verification_labels, predicted_labels)
def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    # https://gist.github.com/ClementC/acf8d5f21fd91c674808
    columnwidth = max([len(x) for x in labels])
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()


# In[22]:


print_cm(cm, classes_labels)


# In[23]:


print("Classification accuracy: %f" %

      metrics.accuracy_score(verification_labels, predicted_labels))


# In[24]:


print("Classification report:\n%s" %
      metrics.classification_report(verification_labels, predicted_labels,
                                    target_names=classes_labels))

