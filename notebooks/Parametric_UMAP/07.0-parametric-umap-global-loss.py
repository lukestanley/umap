#!/usr/bin/env python
# coding: utf-8

# ### Custom embedder for parametric UMAP. 
# This notebook shows you how to run a UMAP projection with a custom embedder. 

# In[2]:


import tensorflow_probability as tfp


# In[3]:


tfp.__version__


# In[4]:


import tensorflow as tf


# In[5]:


tf.__version__


# ### load data

# In[6]:


from tensorflow.keras.datasets import mnist
(train_images, Y_train), (test_images, Y_test) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], -1))/255.
test_images = test_images.reshape((test_images.shape[0], -1))/255.


# ### create parametric umap model

# In[7]:


from umap.parametric_umap import ParametricUMAP


# In[8]:


embedder = ParametricUMAP(
    global_correlation_loss_weight = 0.1, 
    n_epochs=25,
    verbose=True
)


# In[9]:


embedding = embedder.fit_transform(train_images)


# ### plot results

# In[10]:


embedding = embedder.embedding_


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


fig, ax = plt.subplots( figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train.astype(int)[:len(embedding)],
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("UMAP in Tensorflow embedding", fontsize=20)
plt.colorbar(sc, ax=ax);


# ### measure at global structure as correlation of pairwise distances

# In[13]:


import numpy as np
import scipy.stats
import sklearn


# In[14]:


nex = 1000
sample1 = np.random.randint(len(train_images), size=nex)
sample2 = np.random.randint(len(train_images), size=nex)
x1 = train_images[sample1]
x2 = train_images[sample2]
z1 = embedding[sample1]
z2 = embedding[sample2]
x_dist = sklearn.metrics.pairwise_distances(x1, x2).flatten()

z_dist = sklearn.metrics.pairwise_distances(z1, z2).flatten()

corr, p = scipy.stats.pearsonr(x_dist, z_dist)
print("r^2={}, p={}".format(corr, p))


# In[15]:


fig, ax = plt.subplots()

ax.hist(x_dist, color = 'k', alpha = 0.1, density=True)
ax.set_ylabel('Density of data distances')
ax.set_xlabel('Data distance')

ax2 = ax.twinx()
bins = np.linspace(np.min(x_dist), np.max(x_dist), 20)
xbins = np.digitize(x_dist, bins = bins)
zmean = np.array([np.mean(z_dist[xbins == i]) for i in np.unique(xbins)])
zstd = np.array([np.std(z_dist[xbins == i]) for i in np.unique(xbins)])
ax2.plot(bins, zmean)
ax2.fill_between(bins, zmean-zstd, zmean+zstd, alpha = 0.1)
ax2.set_ylabel('Embedding distances')


# ### plotting loss

# In[16]:


embedder._history.keys()


# In[17]:


fig, ax = plt.subplots()
ax.plot(embedder._history['umap_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('Epoch')
ax2 = ax.twinx()
ax2.plot(embedder._history['global_correlation_loss'], color = 'r')
ax2.set_ylabel('global_correlation_loss')


# ## Repeat with more global structure

# ### create parametric umap model

# In[18]:


embedder = ParametricUMAP(
    global_correlation_loss_weight = 1.0,
    n_epochs=25,
    verbose=True
)


# In[19]:


embedding = embedder.fit_transform(train_images)


# ### plot results

# In[20]:


embedding = embedder.embedding_


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


fig, ax = plt.subplots( figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train.astype(int)[:len(embedding)],
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("UMAP in Tensorflow embedding", fontsize=20)
plt.colorbar(sc, ax=ax);


# ### measure at global structure as correlation of pairwise distances

# In[23]:


import numpy as np
import scipy.stats
import sklearn


# In[24]:


nex = 1000
sample1 = np.random.randint(len(train_images), size=nex)
sample2 = np.random.randint(len(train_images), size=nex)
x1 = train_images[sample1]
x2 = train_images[sample2]
z1 = embedding[sample1]
z2 = embedding[sample2]
x_dist = sklearn.metrics.pairwise_distances(x1, x2).flatten()

z_dist = sklearn.metrics.pairwise_distances(z1, z2).flatten()

corr, p = scipy.stats.pearsonr(x_dist, z_dist)
print("r^2={}, p={}".format(corr, p))


# In[25]:


np.unique(xbins), len(np.unique(xbins))


# In[26]:


fig, ax = plt.subplots()

ax.hist(x_dist, color = 'k', alpha = 0.1, density=True)
ax.set_ylabel('Density of data distances')
ax.set_xlabel('Data distance')

ax2 = ax.twinx()
bins = np.linspace(np.min(x_dist), np.max(x_dist), 20)
xbins = np.digitize(x_dist, bins = bins)
zmean = np.array([np.mean(z_dist[xbins == i]) for i in np.unique(xbins)])
zstd = np.array([np.std(z_dist[xbins == i]) for i in np.unique(xbins)])
ax2.plot(np.unique(xbins), zmean)
ax2.fill_between(np.unique(xbins), zmean-zstd, zmean+zstd, alpha = 0.1)
ax2.set_ylabel('Embedding distances')


# ### plotting loss

# In[27]:


embedder._history.keys()


# In[28]:


fig, ax = plt.subplots()
ax.plot(embedder._history['umap_loss'])
ax.set_ylabel('umap_loss')
ax.set_xlabel('Epoch')
ax2 = ax.twinx()
ax2.plot(embedder._history['global_correlation_loss'], color = 'r')
ax2.set_ylabel('global_correlation_loss')


# In[ ]:





# In[ ]:




