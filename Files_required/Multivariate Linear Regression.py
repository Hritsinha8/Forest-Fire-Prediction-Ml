#!/usr/bin/env python
# coding: utf-8

#  NAME : HRITESH SINHA
#  
#  CLASS : 2MScDS-B
#  
#  REG. NO.: 22122151

# **FOREST FIRE PREDICTION MODEL**

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv")
df


# **HANDLING MISSING DATAPOINT**

# In[3]:


df = df.drop(['month', 'day'], axis=1)


# In[4]:


df


# In[5]:


# filling up the missing values with median
import math
median_temp = math.floor(df.temp.median())    # taking only integer value using math.floor() function
median_temp


# In[6]:


df.temp = df.temp.fillna(median_temp)


# In[7]:


df


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df["temp"].value_counts().sum()


# In[11]:


# Import libraries
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize =(10, 7))

# Creating plot
plt.boxplot(df['RH'])

# show plot
plt.show()


# In[12]:


x_df = df.drop('area', axis=1)
y_df = df['area']


# In[13]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)


# In[14]:


model = linear_model.LinearRegression()


# In[15]:


model.fit(X_train, y_train)               
print(model.score(X_train, y_train))


# In[16]:


prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)


# In[17]:


import pickle
pickle.dump(model, open('model.pkl','wb'))


# In[18]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9,7,87,27,100,8.9,5,52,7,0.3]]))

