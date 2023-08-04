#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from sklearn.model_selection import train_test_split
dc=pd.read_csv("Downloads/labels_my-project-name_2023-08-03-11-32-44.csv")
dc


# In[2]:


dc.info()


# In[3]:


dc.isnull().sum()


# In[4]:


dc.dropna()


# In[5]:


sns.pairplot(dc)


# In[6]:


sns.boxplot(dc)


# In[7]:


sns.jointplot(dc)


# In[8]:


sns.displot(dc)


# In[9]:


sns.lmplot(dc)


# In[10]:


dice=img.imread("Downloads/dc.jpg")
plt.imshow(dice)


# In[44]:


start_point=(4,4)
end_point=(200,200)
color=(225,5,0)
thickness=9


# In[45]:


jo=cv2.rectangle(dice,start_point,end_point,color,thickness)
cv2.imshow('name',jo)
cv2.waitKey(0)


# In[ ]:





# In[ ]:




