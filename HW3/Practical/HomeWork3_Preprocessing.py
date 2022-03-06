#!/usr/bin/env python
# coding: utf-8

# In[623]:


import numpy ad np
import pandad ad pd
# ^^^ pyforest auto-imports - don't write above this line
import numpy ad np
import pandad ad pd


# In[624]:


train_data=pd.read_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\Train.csv')
test_data=pd.read_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\Test.csv')


# In[625]:


train_data.drop(["Id"], axis = 1, inplace=True)
test_data.drop(["Id"], axis = 1, inplace=True)


# In[626]:


train_data.info()


# In[627]:


dict(train_data.dtypes)


# ## Fix all numerical columns and null values in categoricals

# In[628]:


#fixing all numeric values
for column in train_data.columns:
    if train_data[column].dtype is np.dtype('O'):
        train_data[column]=np.where(train_data[column].isnull(),"_Unknown_",train_data[column])
    else:
        train_data[column]=np.where(train_data[column].isnull(),train_data[column].mean(),train_data[column])


# In[629]:


#fixing all numeric values
for column in test_data.columns:
    if test_data[column].dtype is np.dtype('O'):
        test_data[column]=np.where(test_data[column].isnull(),"_Unknown_",test_data[column])
    else:
        test_data[column]=np.where(test_data[column].isnull(),train_data[column].mean(),test_data[column])


# In[630]:


train_data.LandSlope.value_counts(dropna=False)


# In[631]:


commons=list(set(train_data.columns) & set(test_data.columns))
len(commons)


# In[632]:


test_data = test_data[commons]
commons.append('SalePrice')
train_data = train_data[commons]
commons


# ## Nominal Variables

# In[633]:


def mapping(orderedlist):
    i=0
    ordered = {}
    for item in orderedlist:
        ordered[str(item)] = i
        i = i+1
    return ordered


# In[634]:


def convert(data,column,mappingList):
    if column not in data.columns:
        return
    if data[column].dtype is np.dtype('O'):
        data[column] = data[column].replace(mapping(mappingList))
        if any(data[column] == '_Unknown_'):
            data[str(column)+'_IsNull'] = np.where(data[column] == '_Unknown_',True,False)
            data[column] = np.where(data[column]=='_Unknown_',-1,data[column])
            data[column] = np.where(data[column]==-1,data[column].mean(),data[column])
            data[str(column)+'_IsNull'] = pd.to_numeric(arg=data[column+'_IsNull'])


# In[635]:


convert(train_data,'LotShape',["Reg", "IR1", "IR2","IR3"])
convert(train_data,'LandContour',["Lvl","Bnk","HLS","Low"])
convert(train_data,'Utilities',["ELO","NoSeWa","NoSewr","AllPub"])
convert(train_data,'LandSlope',["Sev","Mod","Gtl"])
convert(train_data,'ExterQual',["Ex","Gd","TA",'Fa','Po'])
convert(train_data,'ExterCond',["Ex","Gd","TA",'Fa','Po'])
convert(train_data,'BsmtQual',["Ex","Gd","TA",'Fa','Po'])
convert(train_data,'BsmtCond',["Ex","Gd","TA",'Fa','Po'])
convert(train_data,'BsmtExposure',["Gd","Av","Mn",'No','NA'])
convert(train_data,'BsmtFinType1',["GLQ","ALQ","BLQ",'Rec','LwQ','Unf','NA'])
convert(train_data,'BsmtFinType2',["GLQ","ALQ","BLQ",'Rec','LwQ','Unf','NA'])
convert(train_data,'HeatingQC',["Ex","Gd","TA",'Fa','Po'])
convert(train_data,'CentralAir',['No','Yes'])
convert(train_data,'KitchenQual',["Ex","Gd","TA",'Fa','Po'])
convert(train_data,'Functional',["Typ","Min1","Min2",'Mod','Maj1','Maj2','Sev','Sal'])
convert(train_data,'FireplaceQu',["Ex","Gd","TA",'Fa','Po','NA'])
convert(train_data,'GarageFinish',["Fin","RFn","Unf",'NA'])
convert(train_data,'GarageQual',["Ex","Gd","TA",'Fa','Po','NA'])
convert(train_data,'GarageCond',["Ex","Gd","TA",'Fa','Po','NA'])
convert(train_data,'PavedDrive',["Y","P","N"])
convert(train_data,'PoolQC',["Ex","Gd","TA",'Fa','NA'])
convert(train_data,'Fence',["GdPrv","MnPrv","GdWo",'MnWw','NA'])
convert(train_data,'CentralAir',['Y','N'])
train_data


# In[636]:


convert(test_data,'LotShape',["Reg", "IR1", "IR2","IR3"])
convert(test_data,'LandContour',["Lvl","Bnk","HLS","Low"])
convert(test_data,'Utilities',["ELO","NoSeWa","NoSewr","AllPub"])
convert(test_data,'LandSlope',["Sev","Mod","Gtl"])
convert(test_data,'ExterQual',["Ex","Gd","TA",'Fa','Po'])
convert(test_data,'ExterCond',["Ex","Gd","TA",'Fa','Po'])
convert(test_data,'BsmtQual',["Ex","Gd","TA",'Fa','Po'])
convert(test_data,'BsmtCond',["Ex","Gd","TA",'Fa','Po'])
convert(test_data,'BsmtExposure',["Gd","Av","Mn",'No','NA'])
convert(test_data,'BsmtFinType1',["GLQ","ALQ","BLQ",'Rec','LwQ','Unf','NA'])
convert(test_data,'BsmtFinType2',["GLQ","ALQ","BLQ",'Rec','LwQ','Unf','NA'])
convert(test_data,'HeatingQC',["Ex","Gd","TA",'Fa','Po'])
convert(test_data,'CentralAir',['No','Yes'])
convert(test_data,'KitchenQual',["Ex","Gd","TA",'Fa','Po'])
convert(test_data,'Functional',["Typ","Min1","Min2",'Mod','Maj1','Maj2','Sev','Sal'])
convert(test_data,'FireplaceQu',["Ex","Gd","TA",'Fa','Po','NA'])
convert(test_data,'GarageFinish',["Fin","RFn","Unf",'NA'])
convert(test_data,'GarageQual',["Ex","Gd","TA",'Fa','Po','NA'])
convert(test_data,'GarageCond',["Ex","Gd","TA",'Fa','Po','NA'])
convert(test_data,'PavedDrive',["Y","P","N"])
convert(test_data,'PoolQC',["Ex","Gd","TA",'Fa','NA'])
convert(test_data,'Fence',["GdPrv","MnPrv","GdWo",'MnWw','NA'])
convert(test_data,'CentralAir',['Y','N'])
test_data


# In[ ]:





# ## Categorical Variables

# ### Check to see if had null values

# In[637]:


train_data['SaleCondition'].value_counts(dropna=False)


# ### encoding categorical with one-hot-encoding

# In[ ]:





# In[638]:


train_data_new=pd.get_dummies(data=train_data,columns=['MSSubClads','MSZoning','Street',
                                                       'LotConfig','Neighborhood','Condition1',
                                                      'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
                                                    "Alley","MiscFeature" , 'Exterior2nd',
                                                        'MadVnrType',
                                                      'Foundation','Heating',
                                                        'Electrical',
                                                       'GarageType',
                                                       'SaleType','SaleCondition'])
train_data_new


# In[639]:


test_data_new=pd.get_dummies(data=test_data,columns=['MSSubClads','MSZoning','Street',
                                                       'LotConfig','Neighborhood','Condition1',
                                                      'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
                                                      'Exterior2nd',"Alley","MiscFeature",
                                                        'MadVnrType',
                                                      'Foundation','Heating',
                                                        'Electrical',
                                                       'GarageType',
                                                       'SaleType','SaleCondition'])
test_data_new


# In[640]:


train_data_new.isnull().sum().sum()


# In[641]:


columns = list(set(train_data_new.columns)& set(test_data_new.columns))

y = train_data_new.SalePrice

train_copy = train_data_new[columns]
test_copy = test_data_new[columns]

mean = train_copy.mean(axis = 0)
std = train_copy.std(axis = 0)

train_copy -= mean
train_copy /= std

test_copy -= mean
test_copy /= std


# In[ ]:





# In[642]:


importance = {}
for item in columns:
    cor = np.corrcoef(train_copy[item].adtype(float),train_data_new['SalePrice'])[0,1]
    if np.abs(cor)>0.35:
        importance[item]=cor

importantList=list(importance.keys())

test_copy = test_copy[importantList]
# importantList.append('SalePrice')
train_copy = train_copy[importantList]


# In[643]:


train_copy['SalePrice'] = train_data['SalePrice']


# In[644]:


train_copy.shape


# ## Now we can save the data

# In[645]:


train_copy.to_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TrainPreprocessed.csv',index=False)


# In[646]:


test_copy.to_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TestPreprocessed.csv',index=False)


# In[ ]:





# In[ ]:




