#!/usr/bin/env python
# coding: utf-8

# ### Simple UMAP example
# This notebook shows you how to run a UMAP projection on the MNIST dataset, as well as plot the loss, and save the model.

# ### load data

# In[1]:


from tensorflow.keras.datasets import mnist
(train_images, Y_train), (test_images, Y_test) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], -1))/255.
test_images = test_images.reshape((test_images.shape[0], -1))/255.


# ### create parametric umap model

# In[2]:


from umap.parametric_umap import ParametricUMAP


# In[3]:


embedder = ParametricUMAP(n_epochs = 50, verbose=True)


# In[4]:


embedding = embedder.fit_transform(train_images)


# ### plot results

# In[5]:


import matplotlib.pyplot as plt


# In[6]:


fig, ax = plt.subplots( figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train.astype(int),
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("UMAP in Tensorflow embedding", fontsize=20)
plt.colorbar(sc, ax=ax);


# ### plotting loss

# In[7]:


embedder._history.keys()


# In[8]:


fig, ax = plt.subplots()
ax.plot(embedder._history['loss'])
ax.set_ylabel('Cross Entropy')
ax.set_xlabel('Epoch')


# ### train the model for longer

# In[9]:


embedder = ParametricUMAP(n_epochs = 200, verbose=True)

embedding = embedder.fit_transform(train_images)


# In[13]:


fig, ax = plt.subplots( figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train.astype(int),
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("UMAP in Tensorflow embedding", fontsize=20)
plt.colorbar(sc, ax=ax);


# ### saving and loading

# In[10]:


from umap.parametric_umap import load_ParametricUMAP


# In[11]:


embedder.save('/tmp/model')


# In[12]:


embedder = load_ParametricUMAP('/tmp/model')

