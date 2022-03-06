#!/usr/bin/env python
# coding: utf-8

# # CE-40717: Machine Learning

# ## HW6-Gradient Boosting

# ### Installation:
# You can use [sklearn](https://scikit-learn.org) and [xgboost](https://xgboost.readthedocs.io) packages:
# ```python
# !pip install -U scikit-learn
# !pip install xgboost
# ```

# In[ ]:





# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from time import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, plot_confusion_matrix, confusion_matrix


# ### Load & Prepare Dataset:

# In[20]:


np.random.seed(seed=42)

# load dataset:
iris = datasets.load_iris()
X = iris.data
y = iris.target


# preprocess(if you need):


# split dataset to train set and validation set:
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

data_train = xgb.DMatrix(data=x_train, label=y_train)
data_val = xgb.DMatrix(data=x_val,label=y_val)

n_val = 10
class_names = iris.target_names
print(n_val, class_names)


# In[ ]:





# ### Set Hyperparameter for Both Gradine Boost & XGboost:

# In[21]:


# Gradine Boost:
GB_param = dict(n_estimators=5,
                learning_rate=0.01,
                max_depth=3,
                random_state=0)

# XGboost:
XGboost_param = {"eta": 0.3,
                 "silent": True,
                 "objective": "multi:softprob",
                 "num_class": 3,
                 "max_depth": 3}

num_round = 5


# ### Define Classifiers:

# In[31]:


# define classifier for gradient boost:
GB_clf = GradientBoostingClassifier(n_estimators=5,
                learning_rate=0.01,
                max_depth=3,
                random_state=0)



# define classifier for XGboost:
XGboost_clf = xgb.XGBClassifier(eta= 0.3,
                 silent=True,
                 objective="multi:softprob",
                 num_class =3,
                 max_depth= 3)


# ### Train Both Classifiers:

# In[33]:


# train  gradient boost:
tic = time()
trained_GB = GB_clf.fit(x_train,y_train)
toc = time()

# calculate training time for GB:
GB_train_time = toc - tic
print(f"GB_train_time: {1000.0*GB_train_time} millisecond")


# train XGboost:
tic = time()
trained_XGboost =XGboost_clf.fit(x_train,y_train)
toc = time()

# calculate training time for XGboost:
XGboost_train_time = toc - tic
print(f"XGboost_train_time: {1000.0*XGboost_train_time} millisecond")


# ### Prediction on Validation Set:

# In[34]:


# prediction for gradient boost:
tic = time()
y_pred_GB = GB_clf.predict(x_val)
toc = time()

# calculate validation time per data for GB:
GB_val_time_per_data = (toc - tic)/len(x_val) 
print(f"GB_val_time_per_data: {1000.0*GB_val_time_per_data} millisecond")


# prediction for XGboost:
tic = time()
y_pred_XGboost = XGboost_clf.predict(x_val)
toc = time()


# calculate validation time per data for XGboost:
XGboost_val_time_per_data = (toc-tic)/len(x_val)
print(f"XGboost_val_time_per_data: {1000.0*XGboost_val_time_per_data} millisecond")


# ### Evaluation (precision - recall - F1 score - confusion matrix):

# #### for Gradient Boost:

# In[35]:





# In[39]:


# calculate precision
precision_GB = precision_score(y_val,y_pred_GB,average='weighted')

print(f"precision_GB: {precision_GB}")


# In[41]:


# calculate recall
recall_GB = recall_score(y_val,y_pred_GB,average='weighted')

print(f"recall_GB: {recall_GB}")


# In[42]:


# calculate F1 score
f1_GB = f1_score(y_val,y_pred_GB,average='weighted')

print(f"F1_GB: {f1_GB}")


# In[51]:


# calculate confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", "true")]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(XGboost_clf, x_val, y_val,
                                 display_labels=iris.target_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

plt.show()


# #### for XGboost:

# In[44]:


# calculate precision
precision_XGboost = precision_score(y_val,y_pred_XGboost,average='weighted')

print(f"precision_XGboost: {precision_XGboost}")


# In[47]:


# calculate recall
recall_XGboost = recall_score(y_val,y_pred_XGboost,average='weighted')

print(f"recall_XGboost: {recall_XGboost}")


# In[48]:


# calculate F1 score
f1_XGboost = f1_score(y_val,y_pred_XGboost,average='weighted')

print(f"F1_XGboost: {f1_XGboost}")


# In[54]:


# calculate confusion matrix
cm_XGboost = confusion_matrix(y_val,y_pred_XGboost)

print(f"conf_mat_XGboost: {cm_XGboost}")


# ### Compare Gradient Boost & XGboost Algorithm According to Evaluation Part Results:

# Write your analysis here:
# 
# 
# I expected for XGBoost to be better at this task but both of them performed equally well. 

# In[ ]:




