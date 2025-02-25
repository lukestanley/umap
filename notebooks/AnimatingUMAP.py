#!/usr/bin/env python
# coding: utf-8

# # Making Animations of UMAP Hyper-parameters
# 
# Sometimes one of the best ways to see the effects of hyperparameters is simply to visualise what happens as they change. We can do that in practice with UMAP by simply creating an animation that transitions between embeddings generated with variations of hyperparameters. To do this we'll make use of matplotlib and its animation capabilities. Jake Vanderplas has [a great tutorial](https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/) if you want to know more about creating animations with matplotlib.

# **Note:**
# This is a self contained example of how to use UMAP and the impact of individual hyper-parameters. To make sure everything works correctly please use `conda`.
# For install and usage details see [here](https://docs.conda.io/en/latest/miniconda.html)
# 
# To create animations we need `ffmpeg`. It can be installed with `conda`.
# 
# If you already have `ffmpeg` installed on your machine and you know what you are doing you do not need conda. It is only used to install `ffmpeg`.
# 
# => Remove the next two cells if you are not using `conda`.

# In[1]:


get_ipython().system('conda --version')


# In[2]:


get_ipython().system('conda install -c conda-forge ffmpeg -y')


# In[3]:


get_ipython().system('python --version')


# To start we'll need some basic libraries. First ``numpy`` will be needed for basic array manipulation. Since we will be visualising the results we will need ``matplotlib`` and ``seaborn``. Finally we will need ``umap`` for doing the dimension reduction itself.

# In[4]:


get_ipython().system('pip install numpy matplotlib seaborn umap-learn')


# To start let's load everything we'll need

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
from IPython.display import HTML
import seaborn as sns
import itertools
sns.set(style='white', rc={'figure.figsize':(14, 12), 'animation.html': 'html5'})


# In[6]:


# Ignore UserWarnings
import warnings
warnings.simplefilter('ignore', UserWarning)


# In[7]:


from sklearn.datasets import load_digits


# In[8]:


from umap import UMAP


# To try this out we'll needs a reasonably small dataset (so embedding runs don't take *too* long since we'll be doing a lot of them). For ease of reproducibility for everyone else I'll use the digits dataset from sklearn. If you want to try other datasets just drop them in here -- COIL20 might be interesting, or you might have your own data.

# In[9]:


digits = load_digits()
data = digits.data
data


# We need to move the points in between the embeddings given by different parameter values. There are potentially fancy ways to do this (Something using rotation and reflection to get an initial alignment might be interesting), but we'll use straighforward linear interpolation between the two embeddings. To do this we'll need a simple function that can turn out intermediate embeddings for the in-between frames of the animation.

# In[10]:


def tween(e1, e2, n_frames=20):
    for i in range(5):
        yield e1
    for i in range(n_frames):
        alpha = i / float(n_frames - 1)
        yield (1 - alpha) * e1 + alpha * e2
    for i in range(5):
        yield(e2)
    return


# Now that we can fill in intermediate frame we just need to generate all the embeddings. We'll create a function that can take an argument and set of parameter values and then generate all the embeddings including the in-between frames.

# In[11]:


def generate_frame_data(data, arg_name='n_neighbors', arg_list=[]):
    result = []
    es = []
    for arg in arg_list:
        kwargs = {arg_name:arg}
        if len(es) > 0:
            es.append(UMAP(init=es[-1], negative_sample_rate=3, **kwargs).fit_transform(data))
        else:
            es.append(UMAP(negative_sample_rate=3, **kwargs).fit_transform(data))
        
    for e1, e2 in zip(es[:-1], es[1:]):
        result.extend(list(tween(e1, e2)))
        
    return result


# Next we just need to create a function to actually generate the animation given a list of embeddings (one for each frame). This is really just a matter of workign through the details of how matplotlib generates animations -- I would refer you again to Jake's tutorial if you are interested in the detailed mechanics of this.

# In[12]:


def create_animation(frame_data, arg_name='n_neighbors', arg_list=[]):
    fig, ax = plt.subplots()
    all_data = np.vstack(frame_data)
    frame_bounds = (all_data[:, 0].min() * 1.1, 
                    all_data[:, 0].max() * 1.1,
                    all_data[:, 1].min() * 1.1, 
                    all_data[:, 1].max() * 1.1)
    ax.set_xlim(frame_bounds[0], frame_bounds[1])
    ax.set_ylim(frame_bounds[2], frame_bounds[3])
    points = ax.scatter(frame_data[0][:, 0], frame_data[0][:, 1], 
                        s=5, c=digits.target, cmap='Spectral', animated=True)
    title = ax.set_title('', fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(
        points,
        cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05),
        orientation="vertical",
        values=np.arange(10),
        boundaries=np.arange(11)-0.5,
        ticks=np.arange(10),
        drawedges=True,
    )
    cbar.ax.yaxis.set_ticklabels(np.arange(10), fontsize=18)

    def init():
        points.set_offsets(frame_data[0])
        arg = arg_list[0]
        arg_str = f'{arg:.3f}' if isinstance(arg, float) else f'{arg}'
        title.set_text(f'UMAP with {arg_name}={arg_str}')
        return (points,)

    def animate(i):
        points.set_offsets(frame_data[i])
        if (i + 15) % 30 == 0:
            arg = arg_list[(i + 15) // 30]
            arg_str = f'{arg:.3f}' if isinstance(arg, float) else f'{arg}'
            title.set_text(f'UMAP with {arg_name}={arg_str}')
        return (points,)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frame_data), interval=20, blit=True)
    plt.close()
    return anim


# Finally a little bit of glue to make it all go together.

# In[13]:


def animate_param(data, arg_name='n_neighbors', arg_list=[]):
    frame_data = generate_frame_data(data, arg_name, arg_list)
    return create_animation(frame_data, arg_name, arg_list)


# Now we can create an animation. It will be embedded as an HTML5 video into this notebook.

# In[14]:


animate_param(data, 'n_neighbors', [3, 4, 5, 7, 10, 15, 25, 50, 100, 200])


# In[15]:


animate_param(data, 'min_dist', [0.0, 0.01, 0.1, 0.2, 0.4, 0.6, 0.9])


# In[16]:


animate_param(data, 'local_connectivity', [0.1, 0.2, 0.5, 1, 2, 5, 10])


# In[17]:


animate_param(data, 'set_op_mix_ratio', np.linspace(0.0, 1.0, 10))

