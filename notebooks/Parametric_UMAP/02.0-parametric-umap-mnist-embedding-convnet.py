#!/usr/bin/env python
# coding: utf-8

# ### Custom embedder for parametric UMAP. 
# This notebook shows you how to run a UMAP projection with a custom embedder. 

# In[1]:


get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=1')


# ### load data

# In[2]:


from tensorflow.keras.datasets import mnist
(train_images, Y_train), (test_images, Y_test) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], -1))/255.
test_images = test_images.reshape((test_images.shape[0], -1))/255.


# ### define the encoder network

# In[3]:


import tensorflow as tf
dims = (28,28, 1)
n_components = 2
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=dims),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
    ),
    tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=n_components),
])
encoder.summary()


# ### create parametric umap model

# In[4]:


from umap.parametric_umap import ParametricUMAP


# In[5]:


embedder = ParametricUMAP(encoder=encoder, dims=dims, n_components=n_components, n_training_epochs=1, verbose=True)


# In[6]:


embedding = embedder.fit_transform(train_images)


# ### plot results

# In[7]:


embedding = embedder.embedding_


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


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

# In[10]:


embedder._history.keys()


# In[11]:


fig, ax = plt.subplots()
ax.plot(embedder._history['loss'])
ax.set_ylabel('Cross Entropy')
ax.set_xlabel('Epoch')


# In[ ]:




