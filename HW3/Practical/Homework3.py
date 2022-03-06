#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
# ^^^ pyforest auto-imports - don't write above this line
import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TrainPreprocessed.csv')
data_test = pd.read_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TestPreprocessed.csv')


# In[3]:


data


# In[38]:


class GaussianKernels:
    def __init__(self,sigma,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.sigma = sigma

    def calculate_kernel(self,x_i,x_j):
        return 1/(np.sqrt(2*np.pi))*np.exp(-0.5*np.square(np.linalg.norm(x=x_i - x_j,axis=-1))/self.sigma)

    def predict(self,x_test):
        kernels=self.calculate_kernel(self.X_train,x_test)
        if np.sum(kernels)==0:
            return self.Y_train.mean()
        weights = kernels / np.sum(kernels)
        return np.dot(weights,self.Y_train)


# In[39]:


class IndicatorKernels:
    def __init__(self,h,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.h = h

    def calculate_kernel(self,x_i,x_j):
         return np.where(np.abs(np.linalg.norm(x=x_i - x_j,axis=-1)) <= self.h,1,0)

    def predict(self,x_test):
        kernels=self.calculate_kernel(self.X_train,x_test)
        if np.sum(kernels)==0:
            return self.Y_train.mean()
        weights = kernels / np.sum(kernels)
        return np.dot(weights,self.Y_train)


# In[35]:


mask = np.random.rand(len(data)) <= 0.85
train =data[mask]
test = data[~mask]

X_train = train.drop(['SalePrice'],axis=1).to_numpy().astype(np.dtype('float64'))
X_test = test.drop(['SalePrice'],axis=1).to_numpy().astype(np.dtype('float64'))
Y_train = train['SalePrice'].to_numpy().astype(np.dtype('float64'))
Y_test = test['SalePrice'].to_numpy().astype(np.dtype('float64'))


# In[45]:


answers = {}
for h in np.linspace(1,10,300):
    kernel_regressor=IndicatorKernels(h,X_train,Y_train)
    sum = 0
    predicted=[]
    for test in X_test:
        y_prediction=kernel_regressor.predict(test)
        predicted.append(y_prediction)
    
    predicted = np.array(predicted)
    result = np.sqrt(np.mean(np.square(predicted-Y_test)))
    answers[h] = result

plt.figure(figsize=(10,10))
answers_np=np.array(list(answers.items()))
plt.title('Kernel')
plt.xlabel('h')
plt.ylabel('RMSE')
print(np.nanmin(answers_np[:,1]))
plt.plot(answers_np[:,0],answers_np[:,1])

min_indicator = np.argmin(answers_np[:,1])
print(answers_np[min_indicator,:])


# In[46]:


answers = {}
for h in np.linspace(0,5,300):
    kernel_regressor=GaussianKernels(h,X_train,Y_train)
    sum = 0
    predicted=[]
    for test in X_test:
        y_prediction=kernel_regressor.predict(test)
        predicted.append(y_prediction)
    
    predicted = np.array(predicted)
    result = np.sqrt(np.mean(np.square(predicted-Y_test)))
    answers[h] = result

plt.figure(figsize=(10,10))
answers_np2=np.array(list(answers.items()))
plt.title('Kernel')
plt.xlabel('h')
plt.ylabel('RMSE')
print(np.nanmin(answers_np2[:,1]))
plt.plot(answers_np2[:,0],answers_np2[:,1])

min_gaussian = np.nanargmin(answers_np2[:,1])
print(answers_np2[min_gaussian,:])


# In[50]:


kernel_regressor=IndicatorKernels(answers_np[min_indicator,0],X_train,Y_train)
predictionOfIndicator = []
for row in data_test.to_numpy().astype(np.dtype('float64')):
    predictionOfIndicator.append(kernel_regressor.predict(row))
predictionOfIndicatorNumpy = np.array(predictionOfIndicator)


# In[54]:


kernel_gaussian=GaussianKernels(answers_np2[min_gaussian,0],X_train,Y_train)
predictionOfGuassian = []
for row in data_test.to_numpy().astype(np.dtype('float64')):
    predictionOfGuassian.append(kernel_gaussian.predict(row))
predictionOfGaussianNumpy = np.array(predictionOfGuassian)


# ## Plot of test results

# In[68]:


plt.figure(figsize=(10,10))
plt.plot(predictionOfGaussianNumpy)
plt.plot(predictionOfIndicatorNumpy)


# ## plot of best results of gaussian kernel vs actual result

# In[71]:


kernel_regressor=IndicatorKernels(answers_np[min_indicator,0],X_train,Y_train)
predictionOfIndicator = []
for row in X_test:
    predictionOfIndicator.append(kernel_regressor.predict(row))
predictionOfIndicatorNumpy = np.array(predictionOfIndicator)

plt.figure(figsize=(10,10))
plt.plot(predictionOfIndicatorNumpy)
plt.plot(Y_test)


# ## plot of best results of indicator kernel vs actual result

# In[72]:


kernel_gaussian=GaussianKernels(answers_np2[min_gaussian,0],X_train,Y_train)
predictionOfGuassian = []
for row in X_test:
    predictionOfGuassian.append(kernel_gaussian.predict(row))
predictionOfGaussianNumpy = np.array(predictionOfGuassian)

plt.figure(figsize=(10,10))
plt.plot(predictionOfGaussianNumpy)
plt.plot(Y_test)


# In[73]:


np.savetxt("G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TestResultIndicator.csv",predictionOfIndicatorNumpy , delimiter=",")


# In[74]:


np.savetxt("G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TestResultGaussian.csv",predictionOfGaussianNumpy , delimiter=",")


# In[ ]:




