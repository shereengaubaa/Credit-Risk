#!/usr/bin/env python
# coding: utf-8

# ### importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ### loading dataset

# In[2]:


df= pd.read_csv("D:\ds\credit_risk_dataset.csv")
df


# ### dropping duplicates

# In[3]:


df.drop_duplicates(inplace=True)


# ### figuring out missing values

# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.cb_person_default_on_file.value_counts() #ordinal


# In[7]:


df.loan_grade.value_counts() #ordinal


# In[8]:


df.loan_intent.value_counts() #nominal


# In[9]:


df.person_home_ownership.value_counts() #nominal


# In[10]:


sns.distplot(df['person_income'],kde = False, color='red')
plt.show()


# In[11]:


sns.distplot(df['loan_amnt'],kde = False, color='red')
plt.show()


# In[12]:


sns.distplot(df['loan_int_rate'],kde = False, color='red')
plt.show()


# ### outliers

# In[13]:


sns.distplot(df['person_age'],kde = False, color='red')
plt.show()


# In[14]:


df = df.loc[df['person_age']<75, :]


# In[15]:


df.shape


# In[16]:


sns.distplot(df['person_emp_length'],kde = False, color='red')
plt.show()


# In[17]:


df[df['person_emp_length']>60]

#0,216


# In[18]:


df.drop([0,216], axis=0, inplace=True)

df


# In[19]:


df.shape


# In[20]:


df


# In[21]:


#encoding

df.replace({'person_home_ownership': {'RENT':0, 'MORTGAGE':1, 'OWN':2, 'OTHER': 3}}, inplace=True)

df.replace({'loan_intent': {'EDUCATION':0 , 'MEDICAL':1, 'VENTURE':2, 'PERSONAL':3, 'DEBTCONSOLIDATION':4, 'HOMEIMPROVEMENT':5}}, inplace=True)

df.replace({'loan_grade': {'G':0 , 'F':1, 'E':2, 'D':3, 'C':4, 'B':5, 'A':6}}, inplace=True)

df.replace({'cb_person_default_on_file': {'Y':0 , 'N':1}}, inplace=True)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('loan_status', axis=1), df['loan_status'], random_state=0, 
                                                    test_size=0.3, stratify=df['loan_status'], shuffle=True)


# In[24]:


x_train.isnull().sum()


# In[25]:


from sklearn.compose import ColumnTransformer


# ### imputing missing values

# In[26]:




from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[27]:


transf1 = ColumnTransformer( [
    ('impute_person_emp_length' , IterativeImputer(), [3]),
    ('impute_loan_int_rate' , IterativeImputer(), [7])
   
] , remainder='passthrough')


# ### standardization

# In[29]:



from sklearn.preprocessing import MinMaxScaler #minmax when not normally distributed


# In[30]:


transf3 = ColumnTransformer([
    ('scale' , MinMaxScaler(), slice(0,11))
    
])


# In[31]:


#feature selection
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# In[32]:


#transf4 = SelectKBest(score_func=chi2, k=7)


# ###  creating pipelines

# In[33]:


from sklearn.pipeline import Pipeline, make_pipeline


# ### model selection

# In[34]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[35]:



transf5 = RandomForestClassifier()


# In[36]:


pipeln = Pipeline(steps=[
    ('transf1', transf1),
    
    ('transf3', transf3),
   
    ('transf5', transf5)
    
])


# In[37]:


from sklearn import set_config
set_config(display='diagram')


# In[38]:


# training 

pipeln.fit(x_train, y_train)


# In[39]:


x_train


# In[40]:


pipeln.named_steps


# In[41]:


y_pred = pipeln.predict(x_test)


# In[42]:


y_pred


# ### accuracy

# In[43]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[44]:


import pickle


# In[45]:


pickle.dump(pipeln, open('pipeln.pkl', 'wb'))


# In[ ]:




