#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter18/detecting_lanes_in_the_image_of_a_road.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


!wget https://www.dropbox.com/s/vgd22go8a6k721t/road_image.png

# In[2]:


!pip install torch_snippets
from torch_snippets import show, read, subplots, cv2, np
IMG = read('road_image.png')
img = np.uint8(IMG.copy())

# In[3]:


edges = cv2.Canny(img,50,150)
show(edges)

# In[4]:


lines = cv2.HoughLines(edges,1,np.pi/180,150)

# In[5]:


lines = lines[:,0,:]
for rho,theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

show(img)

# In[5]:



