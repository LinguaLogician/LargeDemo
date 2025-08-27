#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter02/Operations_on_tensors.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import torch
x = torch.tensor([[1,2,3,4], [5,6,7,8]]) 
print(x * 10)


# In[ ]:


x = torch.tensor([[1,2,3,4], [5,6,7,8]]) 
y = x.add(10)
print(y)


# In[ ]:


y = torch.tensor([2, 3, 1, 0]) # y.shape == (4)
y = y.view(4,1)                # y.shape == (4, 1)


# In[ ]:


x = torch.randn(10,1,10)
z1 = torch.squeeze(x, 1) # similar to np.squeeze()
# The same operation can be directly performed on
# x by calling squeeze and the dimension to squeeze out
z2 = x.squeeze(1)
assert torch.all(z1 == z2) # all the elements in both tensors are equal
print('Squeeze:\n', x.shape, z1.shape)


# In[ ]:


x = torch.randn(10,10)
print(x.shape)
# torch.size(10,10)
z1 = x.unsqueeze(0)
print(z1.shape)
# torch.size(1,10,10)
# The same can be achieved using [None] indexing
# Adding None will auto create a fake dim at the
# specified axis
x = torch.randn(10,10)
z2, z3, z4 = x[None], x[:,None], x[:,:,None]
print(z2.shape, z3.shape, z4.shape)


# In[ ]:


x = torch.tensor([[1,2,3,4], [5,6,7,8]])
print(torch.matmul(x, y))


# In[ ]:


print(x@y)


# In[ ]:


import torch
x = torch.randn(10,10,10)
z = torch.cat([x,x], axis=0) # np.concatenate()
print('Cat axis 0:', x.shape, z.shape)
# Cat axis 0:  torch.Size([10, 10, 10]) torch.Size([20, 10, 10])
z = torch.cat([x,x], axis=1) # np.concatenate()
print('Cat axis 1:', x.shape, z.shape)
# Cat axis 1: torch.Size([10, 10, 10]) torch.Size([10, 20, 10])


# In[ ]:


x = torch.arange(25).reshape(5,5)
print('Max:', x.shape, x.max()) 


# In[ ]:


x.max(dim=0)


# In[ ]:


m, argm = x.max(dim=1) 
print('Max in axis 1:\n', m, argm) 


# In[ ]:


x = torch.randn(10,20,30)
z = x.permute(2,0,1) # np.permute()
print('Permute dimensions:', x.shape, z.shape)


# In[ ]:


dir(torch.Tensor)


# In[ ]:


help(torch.Tensor.view)


# In[ ]:




