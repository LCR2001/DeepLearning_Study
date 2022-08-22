#!/usr/bin/env python
# coding: utf-8

# - 폐암 수술 환자의 생존율 예측하기 -

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[10]:


import numpy as np
import tensorflow as tf


# In[11]:


np.random.seed(3)
tf.random.set_seed(3)


# In[12]:


Data_set = np.loadtxt("ThoraricSurgery.csv", delimiter=",")


# In[13]:


X = Data_set[:, 0:17]
Y = Data_set[:,17]


# In[14]:


model = Sequential()
model.add(Dense(30,input_dim = 17, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))


# In[16]:


# mean_squared_error = MSE : 오차평균제곱
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, Y, epochs = 100, batch_size = 10)


# In[ ]:




