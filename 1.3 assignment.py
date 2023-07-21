#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
import itertools
import subprocess
from time import time
from scipy import stats
import scipy.optimize as opt  
from scipy.stats import chi2_contingency
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve


# In[38]:


data = pd.read_excel("Approval.xlsx") 
data.shape  

data.head()


# In[39]:


data.shape


# In[40]:


# VERIFYING IF WE IMPORTED THE RIGHT DATASET BY CHECKING THE FIRST 15 ENTRIES OF THE DATA
data.head(10)


# In[41]:


# VERIFYING IF WE IMPORTED THE RIGHT DATASET BY CHECKING THE LAST FIVE ENTRIES OF THE DATA
data.tail()


# In[42]:


# DESCRIPTIVE STATS
data.info()


# In[43]:


data.describe()


# In[44]:


#checking for null values
data.isnull().sum()


# In[47]:


data.fillna(data.mean(), inplace=True)

# For non numeric data using mode
for val in data:
    # Check if the column is of object type
    if data[val].dtypes == 'object':
        # Impute with the most frequent value
        data = data.fillna(data[val].value_counts().index[0])


# In[48]:


data.head(10)


# In[49]:


#Converting all non-numeric data to numeric - using label encoding
from sklearn.preprocessing import LabelEncoder
# Instantiate LabelEncoder
le = LabelEncoder()

for val in data:
    # Compare if the dtype is object
    if data[val].dtypes=='object':
        data[val]=le.fit_transform(data[val])
        
#you can also use one-hot encoding and try building the model


# In[50]:


data.head(10)


# In[51]:


#Converting all non-numeric data to numeric - using one hot encoding
from sklearn.preprocessing import LabelEncoder
# Instantiate LabelEncoder
le = LabelEncoder()

for val in data:
    # Compare if the dtype is object
    if data[val].dtypes=='object':
        data[val]=le.fit_transform(data[val])



# In[52]:


data.head(10)


# In[53]:


fig = plt.figure(figsize=(18,18))
ax = fig.gca()
data.hist(ax=ax, bins = 30)
plt.show()


# In[54]:


data.head(20)


# In[55]:


corr = data.corr()


# In[56]:


fig = plt.figure(figsize=(5,4))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            linewidths=.75)


# In[57]:


from sklearn.model_selection import train_test_split
data = data.drop(['Gender','Citizen'], axis=1) #Other variables which are also not relevant
data = data.values


# In[58]:


# Segregate features and labels into separate variables
X,y = data[:,0:13] , data[:,13]


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)


# In[59]:


from sklearn.preprocessing import MinMaxScaler
# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# you can try to do z-score normalization (look it up!)


# In[60]:


from sklearn.linear_model import LogisticRegression
# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit model to the train set
logreg.fit(rescaledX_train, y_train)


# In[61]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',
random_state=None, solver='warn', tol=0.0001, verbose=0,warm_start=False)


# In[62]:


from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(rescaledX_test)

print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test, y_pred)


# In[63]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = logreg.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



# In[64]:


import statsmodels.formula.api as sm 
import statsmodels.api as sma 
# glm stands for Generalized Linear Model
mylogit = sm.glm( formula = "Approved ~ Debt", 
    data = data, 
    family = sma.families.Binomial() ).fit() 

mylogit.summary()


# In[ ]:




