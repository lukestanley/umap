#!/usr/bin/env python
# coding: utf-8

# ### Adding a custom callback for keras during embedding
# This notebook shows you how to use custom callbacks during training. In this example, we use early stopping to train the network until loss reaches some desired plateau. 

# ### load data

# In[4]:


from tensorflow.keras.datasets import mnist
(train_images, Y_train), (test_images, Y_test) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], -1))/255.
test_images = test_images.reshape((test_images.shape[0], -1))/255.


# ### create parametric umap model

# In[5]:


from umap.parametric_umap import ParametricUMAP


# In[6]:


keras_fit_kwargs = {"callbacks": [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=10**-2,
        patience=10,
        verbose=1,
    )
]}


# In[7]:


embedder = ParametricUMAP(
    verbose=True,
    keras_fit_kwargs = keras_fit_kwargs,
    n_training_epochs=5
)


# In[8]:


embedding = embedder.fit_transform(train_images)


# ### plot results

# In[9]:


embedding = embedder.embedding_


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


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

# In[12]:


embedder._history.keys()


# In[13]:


fig, ax = plt.subplots(figsize=(10,5))
ax.plot(embedder._history['loss'])
ax.set_ylabel('Cross Entropy')
ax.set_xlabel('Epoch')


# In[ ]:




