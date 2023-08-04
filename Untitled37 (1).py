#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets,linear_model
from sklearn.linear_model import LinearRegression


# In[2]:


iris=datasets.load_iris()
print(iris)
mp=pd.DataFrame(iris.data,columns=iris.feature_names)
mp


# In[3]:


mp.info()


# In[4]:


plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("iris")
plt.plot(mp)
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)

# plot:
fig, ax = plt.subplots()

ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 56), yticks=np.linspace(0, 56, 9))

plt.show()


# In[5]:


K=sns.jointplot(data=mp)


# In[6]:


K=sns.boxplot(data=mp)


# In[7]:


K=sns.displot(data=mp)


# In[8]:


K=sns.scatterplot(data=mp)


# In[9]:


K=sns.heatmap(data=mp)


# In[10]:


K=sns.violinplot(data=mp)


# In[11]:


K=sns.kdeplot(data=mp)


# In[12]:


K=sns.countplot(data=mp)


# In[13]:


K=sns.lineplot(data=mp)


# In[14]:


x=mp["sepal length (cm)"].values
y=mp["sepal width (cm)"].values
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
x_train


# In[15]:


x_test


# In[16]:


y_test


# In[17]:


x_train


# In[18]:


y_train


# In[19]:


x_train


# In[21]:


x_train=x_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_train=y_train.reshape(-1,1)
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[20]:


reg.score(x_train,y_train)*100


# In[ ]:




