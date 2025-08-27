#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter02/Auto_gradient_of_tensors.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import numpy as np
import torch
x = torch.tensor([[2., -1.], [1., 1.]], requires_grad=True)
print(x)

# In[ ]:


out = x.pow(2).sum()

# In[ ]:


out.backward()

# In[ ]:


x.grad

# ### Computing gradients for the same case that was present in `Chain_rule.ipynb` notebook in previous chapter

# In[ ]:


x = np.array([[1,1]])
y = np.array([[0]])
x, y = [torch.tensor(i).float() for i in [x,y]]

# In[ ]:


W = [
    np.array([[-0.0053, 0.3793],
              [-0.5820, -0.5204],
              [-0.2723, 0.1896]], dtype=np.float32).T, 
    np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32), 
    np.array([[ 0.1528, -0.1745, -0.1135]], dtype=np.float32).T, 
    np.array([-0.5516], dtype=np.float32)
]

W = [torch.tensor(i, requires_grad=True) for i in W]

# In[ ]:


def feed_forward(inputs, outputs, weights):
    pre_hidden = torch.matmul(inputs,weights[0])+ weights[1]
    hidden = 1/(1+torch.exp(-pre_hidden))
    out = torch.matmul(hidden, weights[2]) + weights[3]
    mean_squared_error = torch.mean(torch.square(out - outputs))
    return mean_squared_error

# In[ ]:


loss = feed_forward(x, y, W)
loss

# In[ ]:


loss.backward()

# In[ ]:


print([w.grad for w in W])

# In[ ]:


updated_W = [w-w.grad for w in W]
updated_W

# In[ ]:



