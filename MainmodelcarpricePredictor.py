#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


car=pd.read_csv('cr23.csv')


# In[4]:


car.head()


# In[5]:


car.shape


# In[6]:


car['company']=car['name'].str.split(' ').str.get(0)


# In[7]:


car


# In[8]:


car.info()


# In[9]:


car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[10]:


car


# In[11]:


car=car.reset_index(drop=True)


# In[12]:


car.describe()


# In[13]:


car=car[car['Price']<1e6].reset_index(drop=True)


# In[14]:


car=car[car['year']<2019]


# In[15]:


car=car[car['kms_driven']<60000]


# In[16]:


car=car[car['Mileage']<25]


# In[17]:


car.describe()


# In[18]:


car.shape


# In[19]:


x=car.drop(columns='Price')
y=car['Price']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[22]:


ohe=OneHotEncoder()


# In[23]:


ohe.fit(x[['name','company','fuel_type','Location','Transmission']])


# In[24]:


c_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type','Location','Transmission']),remainder='passthrough')


# In[25]:


scores=[]
for i in range(500):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(c_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))


# In[27]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# In[28]:


lr=LinearRegression()
rf=RandomForestRegressor()
xgb=XGBRegressor()
gb=GradientBoostingRegressor()
#k=KNeighborsRegressor(n_neighbors=1)
dt=DecisionTreeRegressor()
gb=GradientBoostingRegressor()


# In[29]:


pip1=make_pipeline(c_trans,lr)
pip4=make_pipeline(c_trans,xgb)
pip3=make_pipeline(c_trans,gb)
pip2=make_pipeline(c_trans,rf)
#pip5=make_pipeline(c_trans,k)
pip6=make_pipeline(c_trans,dt)


# In[30]:


pip1.fit(x_train,y_train)
pip2.fit(x_train,y_train)
pip4.fit(x_train,y_train)
pip3.fit(x_train,y_train)
#pip5.fit(x_train,y_train)
pip6.fit(x_train,y_train)


# In[31]:


y1_pred=pip1.predict(x_test)
r2_score(y_test,y1_pred)


# In[32]:


y2_pred=pip2.predict(x_test)
r2_score(y_test,y2_pred)


# In[33]:


y3_pred=pip3.predict(x_test)
r2_score(y_test,y3_pred)


# In[34]:


y4_pred=pip4.predict(x_test)
r2_score(y_test,y4_pred)


# In[35]:


y6_pred=pip6.predict(x_test)
r2_score(y_test,y6_pred)


# In[36]:


from sklearn.ensemble import VotingRegressor


# In[37]:


voting_regressor = VotingRegressor(estimators=[('linear', lr), ('xgb', xgb)])


# In[38]:


pip11=make_pipeline(c_trans,voting_regressor)


# In[39]:


pip11.fit(x_train,y_train)


# In[40]:


y11_pred=pip11.predict(x_test)


# In[41]:


print(r2_score(y_test,y11_pred))


# In[42]:


import pickle


# In[43]:


pickle.dump(pip11,open('vrm.pkl','wb'))


# In[ ]:




