#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter18/detecting_the_number_plate_of_a_car.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


!wget https://raw.githubusercontent.com/zeusees/HyperLPR/master/model/cascade.xml

# In[2]:


!wget https://www.dropbox.com/s/4hbem2kxzqcwo0y/car1.jpg

# In[3]:


!pip install torch_snippets
from torch_snippets import *
plate_cascade = cv2.CascadeClassifier('cascade.xml')
image = read("car1.jpg", 1)

# In[4]:


image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# In[5]:


plates = plate_cascade.detectMultiScale(image_gray, 1.08, 2, minSize=(40, 40),maxSize=(1000, 100))

# In[6]:


image2 = image.astype('uint8')
for (x, y, w, h) in plates:
    print(x,y,w,h)
    x -= w * 0.14
    w += w * 0.75
    y -= h * 0.15
    h += h * 0.3
    cv2.rectangle(image2, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 10)
show(image2, grid=True)

# In[ ]:



