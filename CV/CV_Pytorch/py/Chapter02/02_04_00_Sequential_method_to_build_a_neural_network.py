#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter02/Sequential_method_to_build_a_neural_network.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In[ ]:


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    def __len__(self): 
        return len(self.x)

# In[ ]:


ds = MyDataset(x, y)
dl = DataLoader(ds, batch_size=2, shuffle=True)

# In[ ]:


model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

# In[ ]:


!pip install torch_summary
from torchsummary import summary

# In[ ]:


summary(model, torch.zeros(1,2));

# In[ ]:


loss_func = nn.MSELoss()
from torch.optim import SGD
opt = SGD(model.parameters(), lr = 0.001)
import time
loss_history = []
start = time.time()
for _ in range(50):
    for ix, iy in dl:
        opt.zero_grad()
        loss_value = loss_func(model(ix),iy)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value)
end = time.time()
print(end - start)

# In[ ]:


val = [[8,9],[10,11],[1.5,2.5]]
val = torch.tensor(val).float()

# In[ ]:


model(val.to(device))

# In[ ]:


val.sum(-1)

# In[ ]:



