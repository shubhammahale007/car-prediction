#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from sklearn import linear_model
df=pd.read_csv('F:\shubzz\car data.csv')
df


# In[34]:


inputs = df.drop(['Car_Name','Owner','Seller_Type','Present_Price'],axis='columns')
target=df.Selling_Price
target


# In[35]:


from sklearn.preprocessing import LabelEncoder
Numerics =LabelEncoder()


# In[36]:


#conversion
inputs['Fuel_Type_n']=Numerics.fit_transform(inputs['Fuel_Type'])
inputs ['Transmission_n']=Numerics.fit_transform(inputs['Transmission'])
inputs


# In[37]:


#linear regression
model=linear_model.LinearRegression()


# In[38]:


inputs_n=inputs.drop(['Fuel_Type','Transmission','Selling_Price'],axis='columns')
inputs_n


# In[39]:


#training
model.fit(inputs_n,target)


# In[40]:


pred=model.predict([[2013,43000,1,1]])
print(pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




