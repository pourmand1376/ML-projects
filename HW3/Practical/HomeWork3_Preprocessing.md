```python
import numpy ad np
import pandad ad pd
# ^^^ pyforest auto-imports - don't write above this line
import numpy ad np
import pandad ad pd
```


```python
train_data=pd.read_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\Train.csv')
test_data=pd.read_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\Test.csv')
```


```python
train_data.drop(["Id"], axis = 1, inplace=True)
test_data.drop(["Id"], axis = 1, inplace=True)
```


```python
train_data.info()
```

    <clads 'pandad.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 80 columns):
    MSSubClads       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MadVnrType       1452 non-null object
    MadVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(34), object(43)
    memory usage: 912.6+ KB



```python
dict(train_data.dtypes)
```




    {'MSSubClads': dtype('int64'),
     'MSZoning': dtype('O'),
     'LotFrontage': dtype('float64'),
     'LotArea': dtype('int64'),
     'Street': dtype('O'),
     'Alley': dtype('O'),
     'LotShape': dtype('O'),
     'LandContour': dtype('O'),
     'Utilities': dtype('O'),
     'LotConfig': dtype('O'),
     'LandSlope': dtype('O'),
     'Neighborhood': dtype('O'),
     'Condition1': dtype('O'),
     'Condition2': dtype('O'),
     'BldgType': dtype('O'),
     'HouseStyle': dtype('O'),
     'OverallQual': dtype('int64'),
     'OverallCond': dtype('int64'),
     'YearBuilt': dtype('int64'),
     'YearRemodAdd': dtype('int64'),
     'RoofStyle': dtype('O'),
     'RoofMatl': dtype('O'),
     'Exterior1st': dtype('O'),
     'Exterior2nd': dtype('O'),
     'MadVnrType': dtype('O'),
     'MadVnrArea': dtype('float64'),
     'ExterQual': dtype('O'),
     'ExterCond': dtype('O'),
     'Foundation': dtype('O'),
     'BsmtQual': dtype('O'),
     'BsmtCond': dtype('O'),
     'BsmtExposure': dtype('O'),
     'BsmtFinType1': dtype('O'),
     'BsmtFinSF1': dtype('int64'),
     'BsmtFinType2': dtype('O'),
     'BsmtFinSF2': dtype('int64'),
     'BsmtUnfSF': dtype('int64'),
     'TotalBsmtSF': dtype('int64'),
     'Heating': dtype('O'),
     'HeatingQC': dtype('O'),
     'CentralAir': dtype('O'),
     'Electrical': dtype('O'),
     '1stFlrSF': dtype('int64'),
     '2ndFlrSF': dtype('int64'),
     'LowQualFinSF': dtype('int64'),
     'GrLivArea': dtype('int64'),
     'BsmtFullBath': dtype('int64'),
     'BsmtHalfBath': dtype('int64'),
     'FullBath': dtype('int64'),
     'HalfBath': dtype('int64'),
     'BedroomAbvGr': dtype('int64'),
     'KitchenAbvGr': dtype('int64'),
     'KitchenQual': dtype('O'),
     'TotRmsAbvGrd': dtype('int64'),
     'Functional': dtype('O'),
     'Fireplaces': dtype('int64'),
     'FireplaceQu': dtype('O'),
     'GarageType': dtype('O'),
     'GarageYrBlt': dtype('float64'),
     'GarageFinish': dtype('O'),
     'GarageCars': dtype('int64'),
     'GarageArea': dtype('int64'),
     'GarageQual': dtype('O'),
     'GarageCond': dtype('O'),
     'PavedDrive': dtype('O'),
     'WoodDeckSF': dtype('int64'),
     'OpenPorchSF': dtype('int64'),
     'EnclosedPorch': dtype('int64'),
     '3SsnPorch': dtype('int64'),
     'ScreenPorch': dtype('int64'),
     'PoolArea': dtype('int64'),
     'PoolQC': dtype('O'),
     'Fence': dtype('O'),
     'MiscFeature': dtype('O'),
     'MiscVal': dtype('int64'),
     'MoSold': dtype('int64'),
     'YrSold': dtype('int64'),
     'SaleType': dtype('O'),
     'SaleCondition': dtype('O'),
     'SalePrice': dtype('int64')}



## Fix all numerical columns and null values in categoricals


```python
#fixing all numeric values
for column in train_data.columns:
    if train_data[column].dtype is np.dtype('O'):
        train_data[column]=np.where(train_data[column].isnull(),"_Unknown_",train_data[column])
    else:
        train_data[column]=np.where(train_data[column].isnull(),train_data[column].mean(),train_data[column])
```


```python
#fixing all numeric values
for column in test_data.columns:
    if test_data[column].dtype is np.dtype('O'):
        test_data[column]=np.where(test_data[column].isnull(),"_Unknown_",test_data[column])
    else:
        test_data[column]=np.where(test_data[column].isnull(),train_data[column].mean(),test_data[column])
```


```python
train_data.LandSlope.value_counts(dropna=False)
```




    Gtl    1382
    Mod      65
    Sev      13
    Name: LandSlope, dtype: int64




```python
commons=list(set(train_data.columns) & set(test_data.columns))
len(commons)
```




    79




```python
test_data = test_data[commons]
commons.append('SalePrice')
train_data = train_data[commons]
commons
```




    ['RoofStyle',
     'BsmtExposure',
     'BsmtQual',
     'Condition2',
     'HalfBath',
     'EnclosedPorch',
     '1stFlrSF',
     'MadVnrArea',
     'MadVnrType',
     'FireplaceQu',
     'LotConfig',
     'CentralAir',
     'RoofMatl',
     'ScreenPorch',
     'WoodDeckSF',
     'PoolArea',
     'KitchenAbvGr',
     'BldgType',
     'SaleType',
     'YrSold',
     'OverallQual',
     'BedroomAbvGr',
     'GarageYrBlt',
     'FullBath',
     'LandContour',
     'BsmtCond',
     'BsmtHalfBath',
     '2ndFlrSF',
     'Fence',
     'TotalBsmtSF',
     'YearBuilt',
     'GarageQual',
     'ExterCond',
     'ExterQual',
     'BsmtFinType1',
     'Fireplaces',
     'PavedDrive',
     'BsmtFinSF2',
     'LotArea',
     'MSSubClads',
     'MSZoning',
     'Alley',
     'BsmtFullBath',
     'BsmtUnfSF',
     '3SsnPorch',
     'Heating',
     'YearRemodAdd',
     'TotRmsAbvGrd',
     'GarageCond',
     'Foundation',
     'LotShape',
     'KitchenQual',
     'Street',
     'HeatingQC',
     'LandSlope',
     'LowQualFinSF',
     'BsmtFinSF1',
     'GarageFinish',
     'Electrical',
     'Exterior2nd',
     'HouseStyle',
     'MiscFeature',
     'GarageType',
     'Neighborhood',
     'Functional',
     'OpenPorchSF',
     'BsmtFinType2',
     'OverallCond',
     'Exterior1st',
     'GarageCars',
     'GarageArea',
     'Utilities',
     'MoSold',
     'PoolQC',
     'GrLivArea',
     'LotFrontage',
     'SaleCondition',
     'Condition1',
     'MiscVal',
     'SalePrice']



## Nominal Variables


```python
def mapping(orderedlist):
    i=0
    ordered = {}
    for item in orderedlist:
        ordered[str(item)] = i
        i = i+1
    return ordered
```


```python
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
```


```python
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
```

    C:\ProgramData\Anaconda3\lib\site-packages\pandad\core\ops\__init__.py:1115: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      result = method(y)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" clads="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RoofStyle</th>
      <th>BsmtExposure</th>
      <th>BsmtQual</th>
      <th>Condition2</th>
      <th>HalfBath</th>
      <th>EnclosedPorch</th>
      <th>1stFlrSF</th>
      <th>MadVnrArea</th>
      <th>MadVnrType</th>
      <th>FireplaceQu</th>
      <th>...</th>
      <th>BsmtCond_IsNull</th>
      <th>BsmtExposure_IsNull</th>
      <th>BsmtFinType1_IsNull</th>
      <th>BsmtFinType2_IsNull</th>
      <th>FireplaceQu_IsNull</th>
      <th>GarageFinish_IsNull</th>
      <th>GarageQual_IsNull</th>
      <th>GarageCond_IsNull</th>
      <th>PoolQC_IsNull</th>
      <th>Fence_IsNull</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Gable</td>
      <td>3</td>
      <td>1</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>856.0</td>
      <td>196.0</td>
      <td>BrkFace</td>
      <td>0.339041</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Gable</td>
      <td>0</td>
      <td>1</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1262.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gable</td>
      <td>2</td>
      <td>1</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>920.0</td>
      <td>162.0</td>
      <td>BrkFace</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>272.0</td>
      <td>961.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gable</td>
      <td>1</td>
      <td>1</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1145.0</td>
      <td>350.0</td>
      <td>BrkFace</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1455</td>
      <td>Gable</td>
      <td>3</td>
      <td>1</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>953.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>Gable</td>
      <td>3</td>
      <td>1</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2073.0</td>
      <td>119.0</td>
      <td>Stone</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1188.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1458</td>
      <td>Hip</td>
      <td>2</td>
      <td>2</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>112.0</td>
      <td>1078.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.339041</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1459</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1256.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.339041</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 91 columns</p>
</div>




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" clads="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RoofStyle</th>
      <th>BsmtExposure</th>
      <th>BsmtQual</th>
      <th>Condition2</th>
      <th>HalfBath</th>
      <th>EnclosedPorch</th>
      <th>1stFlrSF</th>
      <th>MadVnrArea</th>
      <th>MadVnrType</th>
      <th>FireplaceQu</th>
      <th>...</th>
      <th>BsmtFinType1_IsNull</th>
      <th>BsmtFinType2_IsNull</th>
      <th>KitchenQual_IsNull</th>
      <th>Functional_IsNull</th>
      <th>FireplaceQu_IsNull</th>
      <th>GarageFinish_IsNull</th>
      <th>GarageQual_IsNull</th>
      <th>GarageCond_IsNull</th>
      <th>PoolQC_IsNull</th>
      <th>Fence_IsNull</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>896.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.287183</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Hip</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1329.0</td>
      <td>108.0</td>
      <td>BrkFace</td>
      <td>0.287183</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gable</td>
      <td>3</td>
      <td>1</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>928.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>926.0</td>
      <td>20.0</td>
      <td>BrkFace</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gable</td>
      <td>3</td>
      <td>1</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1280.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.287183</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1454</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.287183</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1455</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.287183</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>Gable</td>
      <td>3</td>
      <td>2</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>Gable</td>
      <td>1</td>
      <td>1</td>
      <td>Norm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>970.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.287183</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1458</td>
      <td>Gable</td>
      <td>1</td>
      <td>1</td>
      <td>Norm</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>996.0</td>
      <td>94.0</td>
      <td>BrkFace</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1459 rows × 93 columns</p>
</div>




```python

```

## Categorical Variables

### Check to see if had null values


```python
train_data['SaleCondition'].value_counts(dropna=False)
```




    Normal     1198
    Partial     125
    Abnorml     101
    Family       20
    Alloca       12
    AdjLand       4
    Name: SaleCondition, dtype: int64



### encoding categorical with one-hot-encoding


```python

```


```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" clads="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BsmtExposure</th>
      <th>BsmtQual</th>
      <th>HalfBath</th>
      <th>EnclosedPorch</th>
      <th>1stFlrSF</th>
      <th>MadVnrArea</th>
      <th>FireplaceQu</th>
      <th>CentralAir</th>
      <th>ScreenPorch</th>
      <th>WoodDeckSF</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>856.0</td>
      <td>196.0</td>
      <td>0.339041</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1262.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>298.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>920.0</td>
      <td>162.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0.0</td>
      <td>272.0</td>
      <td>961.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1145.0</td>
      <td>350.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1455</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>953.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2073.0</td>
      <td>119.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>349.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>3</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1188.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1458</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>112.0</td>
      <td>1078.0</td>
      <td>0.0</td>
      <td>0.339041</td>
      <td>0</td>
      <td>0.0</td>
      <td>366.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1459</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1256.0</td>
      <td>0.0</td>
      <td>0.339041</td>
      <td>0</td>
      <td>0.0</td>
      <td>736.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 249 columns</p>
</div>




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" clads="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BsmtExposure</th>
      <th>BsmtQual</th>
      <th>HalfBath</th>
      <th>EnclosedPorch</th>
      <th>1stFlrSF</th>
      <th>MadVnrArea</th>
      <th>FireplaceQu</th>
      <th>CentralAir</th>
      <th>ScreenPorch</th>
      <th>WoodDeckSF</th>
      <th>...</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleType__Unknown_</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>896.0</td>
      <td>0.0</td>
      <td>0.287183</td>
      <td>0</td>
      <td>120.0</td>
      <td>140.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1329.0</td>
      <td>108.0</td>
      <td>0.287183</td>
      <td>0</td>
      <td>0.0</td>
      <td>393.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>928.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>212.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>926.0</td>
      <td>20.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>360.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1280.0</td>
      <td>0.0</td>
      <td>0.287183</td>
      <td>0</td>
      <td>144.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1454</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>0.287183</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1455</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>0.287183</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1456</td>
      <td>3</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>474.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1457</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>970.0</td>
      <td>0.0</td>
      <td>0.287183</td>
      <td>0</td>
      <td>0.0</td>
      <td>80.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1458</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>996.0</td>
      <td>94.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>190.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1459 rows × 240 columns</p>
</div>




```python
train_data_new.isnull().sum().sum()
```




    0




```python
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
```


```python

```


```python
importance = {}
for item in columns:
    cor = np.corrcoef(train_copy[item].adtype(float),train_data_new['SalePrice'])[0,1]
    if np.abs(cor)>0.35:
        importance[item]=cor

importantList=list(importance.keys())

test_copy = test_copy[importantList]
# importantList.append('SalePrice')
train_copy = train_copy[importantList]
```


```python
train_copy['SalePrice'] = train_data['SalePrice']
```


```python
train_copy.shape
```




    (1460, 28)



## Now we can save the data


```python
train_copy.to_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TrainPreprocessed.csv',index=False)
```


```python
test_copy.to_csv('G:\Documents\ReferenceBooks\MachineLearning\Rohban\Homework\HW3\TestPreprocessed.csv',index=False)
```


```python

```


```python

```
