#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter18/detecting_objects_based_on_color.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


!wget https://www.dropbox.com/s/utrkdooh08y9mvm/uno_card.png
!pip install torch_snippets

# In[2]:


from torch_snippets import *
import cv2, numpy as np

# In[4]:


img = read('uno_card.png', 1)
show(img)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# In[5]:


lower_green = np.array([45,100,100])
upper_green = np.array([80,255,255])

# In[6]:


mask = cv2.inRange(hsv, lower_green, upper_green)

# In[7]:


res = cv2.bitwise_and(img, img, mask=mask)

# In[8]:


subplots([img, mask, res], nc=3, figsize=(10,5), titles=['Original image','Mask on image','Resulting image'])

# In[ ]:



