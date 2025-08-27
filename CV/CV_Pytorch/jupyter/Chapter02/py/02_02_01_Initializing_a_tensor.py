#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter02/Initializing_a_tensor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import torch
x = torch.tensor([[1,2]])
y = torch.tensor([[1],[2]])


# In[ ]:


print(x.shape)
# torch.Size([1,2]) # one entity of two items
print(y.shape)
# torch.Size([2,1]) # two entities of one item each
print(x.dtype)
# torch.int64


# In[ ]:


x = torch.tensor([False, 1, 2.0])
print(x)
# tensor([0., 1., 2.])


# In[ ]:


torch.zeros((3, 4))


# In[ ]:


torch.ones((3, 4))


# In[ ]:


torch.randint(low=0, high=10, size=(3,4))


# In[ ]:


torch.rand(3, 4)


# In[ ]:


torch.randn((3,4))


# In[ ]:


import numpy as np
x = np.array([[10,20,30],[2,3,4]])
y = torch.tensor(x)
print(type(x), type(y))


# In[ ]:




