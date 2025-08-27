#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter02/Numpy_Vs_Torch_object_computation_speed_comparison.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import torch
x = torch.rand(1, 6400)
y = torch.rand(6400, 5000)

# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', "This exercise assumes the notebook is on a GPU machine"

# In[3]:


x, y = x.to(device), y.to(device)

# In[4]:


%timeit z=(x@y)

# In[5]:


x, y = x.cpu(), y.cpu()
%timeit z=(x@y)

# In[6]:


import numpy as np
x = np.random.random((1, 6400))
y = np.random.random((6400, 5000))
%timeit z = np.matmul(x,y)

# In[ ]:



