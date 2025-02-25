#!/usr/bin/env python
# coding: utf-8

# ### Reconstruction with a custom network. 
# This notebook extends the last notebook to simultaneously train a decoder network, which translates from embedding back into dataspace. It also shows you how to use validation data for the reconstruction network during training.

# ### load data

# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


from tensorflow.keras.datasets import mnist
(train_images, Y_train), (test_images, Y_test) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], -1))/255.
test_images = test_images.reshape((test_images.shape[0], -1))/255.


# ### define the encoder network

# In[4]:


import tensorflow as tf


# In[5]:


tf.__version__


# In[6]:


tf.config.list_physical_devices('GPU')


# In[7]:


dims = (28,28, 1)
n_components = 2
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=dims),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=n_components),
])
encoder.summary()


# In[8]:


decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(n_components)),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=7 * 7 * 128, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 128)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    )
])
decoder.summary()


# ### create parametric umap model

# In[9]:


from umap.parametric_umap import ParametricUMAP


# In[10]:


embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    dims=dims,
    n_components=n_components,
    n_training_epochs=1, # dicates how many total training epochs to run
    n_epochs = 50, # dicates how many times edges are trained per 'epoch' to keep consistent with non-parametric UMAP
    parametric_reconstruction= True,
    reconstruction_validation=test_images,
    parametric_reconstruction_loss_fcn = tf.keras.losses.MSE,
    verbose=True,
)


# In[11]:


train_images.shape


# In[12]:


embedding = embedder.fit_transform(train_images)


# ### plot reconstructions

# In[13]:


test_images_recon = embedder.inverse_transform(embedder.transform(test_images.reshape(len(test_images), 28,28,1)))


# In[14]:


import numpy as np
import matplotlib.pyplot as plt


# In[15]:


np.min(test_images), np.max(test_images)


# In[16]:


nex = 10
fig, axs = plt.subplots(ncols=10, nrows=2, figsize=(nex, 2))
for i in range(nex):
    axs[0, i].matshow(np.squeeze(test_images[i].reshape(28, 28, 1)), cmap=plt.cm.Greys)
    axs[1, i].matshow(
        np.squeeze(test_images_recon[i].reshape(28, 28, 1)),
        cmap=plt.cm.Greys, vmin = 0, vmax = 1
    )
for ax in axs.flatten():
    ax.axis("off")


# ### plot results

# In[17]:


embedding = embedder.embedding_


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


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

# In[20]:


embedder._history.keys()


# In[21]:


fig, axs = plt.subplots(ncols=2, figsize=(10,5))
ax = axs[0]
ax.plot(embedder._history['loss'])
ax.set_ylabel('Cross Entropy')
ax.set_xlabel('Epoch')

ax = axs[1]
ax.plot(embedder._history['reconstruction_loss'], label='train')
ax.plot(embedder._history['val_reconstruction_loss'], label='valid')
ax.legend()
ax.set_ylabel('Cross Entropy')
ax.set_xlabel('Epoch')


# In[ ]:




