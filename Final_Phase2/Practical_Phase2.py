#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Practical Phase1 
## Amir Pourmand
## Stu No: 99210259


# # Downloading Required Dataset

# In[1]:


get_ipython().system('gdown --id 15JJ6ZysFM57tlUjXo2nHVhkGwePbVMVV -O dataset_first.csv')


# In[2]:


get_ipython().system('gdown --id 1uykBJxWH5v5BsSuuwM0r9WLiKWQrDiDJ -O dataset_tune.csv')


# In[3]:


import pandas as pd
dataset = pd.read_csv('dataset_first.csv')
dataset_tune = pd.read_csv('dataset_tune.csv')


# In[4]:


# Load the Drive helper and mount
from google.colab import drive
drive.mount('/content/drive')


# In[5]:


import scipy.sparse
import numpy
import pandas as pd

X_train_2_BOW=scipy.sparse.load_npz('/content/drive/MyDrive/DataForColob/ML_Project/X_train_2_BOW.npz')
X_test_2_BOW=scipy.sparse.load_npz('/content/drive/MyDrive/DataForColob/ML_Project/X_test_2_BOW.npz')

X_train_w2v=pd.read_pickle('/content/drive/MyDrive/DataForColob/ML_Project/X_train_w2v.pkl')
X_test_w2v=pd.read_pickle('/content/drive/MyDrive/DataForColob/ML_Project/X_test_w2v.pkl')

y_train = numpy.load('/content/drive/MyDrive/DataForColob/ML_Project/y_train.npy')
y_test = numpy.load('/content/drive/MyDrive/DataForColob/ML_Project/y_test.npy')

import pickle
svm_w2v = pickle.load(open('/content/drive/MyDrive/DataForColob/ML_Project/SVM.pkl', 'rb'))
knn_w2v = pickle.load(open('/content/drive/MyDrive/DataForColob/ML_Project/KNN.pkl', 'rb'))
lr_w2v = pickle.load(open('/content/drive/MyDrive/DataForColob/ML_Project/LR.pkl', 'rb'))
mlp_best = pickle.load(open('/content/drive/MyDrive/DataForColob/ML_Project/best.pkl', 'rb'))
vectorizer_tfidf=pickle.load(open('/content/drive/MyDrive/DataForColob/ML_Project/vectorizer.pkl', 'rb'))

X_w2v = list(X_train_w2v)
X_w2v.extend(X_test_w2v )
len(X_w2v)


# In[6]:


import numpy as np
y_total = np.concatenate([y_train,y_test])
X_bow = scipy.sparse.vstack([X_train_2_BOW,X_test_2_BOW])


# In[7]:


lr_w2v.score(list(X_test_w2v),y_test)


# # Imports 

# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# # Clustering 

# ## PCA 

# In[9]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca_w2v=pca.fit_transform(X_w2v)


# ## SVD

# In[10]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2, n_iter=7)
svd_bow=svd.fit_transform(X_bow)
svd_bow.shape


# ## K-Means 

# In[12]:


import matplotlib.pyplot as plt
def plot_scatter(X,pred):
    u_labels = np.unique(pred)
    for i in u_labels:
        plt.scatter(X[pred==i,0],X[pred==i,1],label=i)
    plt.legend()
    plt.show()


# In[ ]:


from sklearn.cluster import KMeans

for k in range(2,6):
    kmeans = KMeans(n_clusters=k)
    kmeans_label=kmeans.fit_predict(pca_w2v)
    plot_scatter(pca_w2v,kmeans_label)


# In[13]:


from sklearn.cluster import KMeans

for k in range(2,6):
    kmeans = KMeans(n_clusters=k)
    kmeans_label=kmeans.fit_predict(svd_bow)
    plot_scatter(svd_bow,kmeans_label)


# ## GMM

# In[ ]:


from sklearn.mixture import GaussianMixture

for k in range(2,6):
    gm = GaussianMixture(n_components=k)
    gm_pred=gm.fit_predict(pca_w2v)
    plot_scatter(pca_w2v,gm_pred)


# In[14]:


from sklearn.mixture import GaussianMixture

for k in range(2,6):
    gm = GaussianMixture(n_components=k)
    gm_pred=gm.fit_predict(svd_bow)
    plot_scatter(svd_bow,gm_pred)


# ## Agglomorative 

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

max_data= 30000
for k in range(2,6):
    agg = AgglomerativeClustering(n_clusters=k)
    agg_pred=agg.fit_predict(pca_w2v[:max_data])
    plot_scatter(pca_w2v[:max_data],agg_pred)


# In[18]:


from sklearn.cluster import AgglomerativeClustering

max_data= 30000
for k in range(2,6):
    agg = AgglomerativeClustering(n_clusters=k)
    agg_pred=agg.fit_predict(svd_bow[:max_data])
    plot_scatter(svd_bow[:max_data],agg_pred)


# ## Comparsion

# In[16]:


from sklearn import metrics

def get_analysis(name,true_label,predicted_label):
    print('V Measure ', name, ':', metrics.v_measure_score(true_label,predicted_label))
    print('Adjusted RandScore Measure ', name, ':', metrics.adjusted_rand_score(true_label,predicted_label))
    print('Adjusted Mutual Information ', name, ':', metrics.adjusted_mutual_info_score(true_label,predicted_label))
    print('Homogenity', name, ':', metrics.homogeneity_score(true_label,predicted_label))
    print('-'*30)


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

kmeans = KMeans(n_clusters=2)
kmeans_label=kmeans.fit_predict(pca_w2v)

gm = GaussianMixture(n_components=2)
gm_pred=gm.fit_predict(pca_w2v)

max_data = 30000
agg = AgglomerativeClustering(n_clusters=2)
agg_pred=agg.fit_predict(pca_w2v[:max_data])

get_analysis('kmeans',y_total,kmeans_label)
get_analysis('gm', y_total,gm_pred)
get_analysis('agg',y_total[:max_data],agg_pred)


# In[17]:


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

kmeans = KMeans(n_clusters=2)
kmeans_label=kmeans.fit_predict(svd_bow)

gm = GaussianMixture(n_components=2)
gm_pred=gm.fit_predict(svd_bow)

max_data = 30000
agg = AgglomerativeClustering(n_clusters=2)
agg_pred=agg.fit_predict(svd_bow[:max_data])

get_analysis('kmeans',y_total,kmeans_label)
get_analysis('gm', y_total,gm_pred)
get_analysis('agg',y_total[:max_data],agg_pred)


# ## Semantic Comparison

# In[ ]:


gm = GaussianMixture(n_components=3)
gm_pred=gm.fit_predict(pca_w2v)
for i in range(3):
    print(list(dataset[gm_pred==i][2:3]['sentiment']))
for i in range(3):
    print(list(dataset[gm_pred==i][2:3]['comment']))


# In[ ]:


# first one - very negative
# second one: very positive
# third one: good but not very complimentary


# # Fine Tuning

# ## Initial Run on MLP

# In[189]:


get_ipython().system('pip install contractions')
get_ipython().system('pip install unidecode')
get_ipython().system('pip install word2number')

import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split


#for bag of words
from sklearn.feature_extraction.text import CountVectorizer


#these are all for preprocessing
import nltk
from nltk.tokenize import word_tokenize
import re
from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# this is required for word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[190]:


def remove_all_non_alphabetic(text):
  return re.sub('[^A-Za-z]',' ',text)

def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

stop_words = set(stopwords.words('english'))
def remove_stop_words(token):
  return [item for item in token if item not in stop_words]

lemma = WordNetLemmatizer()
def lemmatization(token):
  return [lemma.lemmatize(word=w,pos='v') for w in token]

def clean_length(token):
  return [item for item in token if len(item)>2]

def punctuation_removal(text):
    return re.sub(r'[\.\?\!\,\:\;\"]', '', text)

def text_merge(token):
  return ' '.join([i for i in token if not i.isdigit()])


# In[191]:


def process_level1(data):
    return (data.apply(str.lower)
                .apply(remove_all_non_alphabetic)
                .apply(word_tokenize)
                .apply(text_merge))

def process_level2(data):
    return (data.apply(str.lower)
        .apply(contractions.fix)
        .apply(strip_html_tags)
        .apply(remove_accented_chars)
        .apply(remove_all_non_alphabetic)
        .apply(word_tokenize)
        .apply(remove_stop_words)
        .apply(lemmatization)
        .apply(clean_length)
        .apply(text_merge))


# In[192]:


X_train_small,X_test_small,y_train_small,y_test_small=train_test_split(dataset_tune['comment'],
                                                                       dataset_tune['sentiment'],test_size=0.2)

X_train_small = process_level2(X_train_small)
X_test_small = process_level2(X_test_small)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer( min_df=0.01,max_df=0.5)
X_train_small_tfidf=vectorizer.fit_transform(X_train_small)
X_test_small_tfidf = vectorizer.transform(X_test_small)


# In[193]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
def print_confusion_matrix(y_test,y_prediction,title):
    print(classification_report(y_test,y_prediction))
    matrix = cm(y_test,y_prediction)
    fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                    show_absolute=True,
                                    show_normed=True,
                                    colorbar=True)
    plt.title(title)
    plt.show()


# In[194]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

grid_params = {
    'hidden_layer_sizes':[(250),(100),(90),(40,10),(50,10)]
}

mlp = MLPClassifier(learning_rate='adaptive',solver='adam',max_iter=1000)

mlp_cv = GridSearchCV(estimator=mlp,param_grid=grid_params,cv=2)

mlp_cv.fit(X_train_small_tfidf,y_train_small)
mlp_prediction=mlp_cv.predict(X_test_small_tfidf)
print_confusion_matrix(y_test_small,mlp_prediction,'TFIDF: MLP ')

display(pd.DataFrame( mlp_cv.cv_results_))


# ## Fine tune based on previous model

# In[197]:


X_train_small_tfidf_olddata=vectorizer_tfidf.transform(X_train_small)
X_test_small_tfidf_olddata = vectorizer_tfidf.transform(X_test_small)

mlp_best = MLPClassifier(warm_start=True)
mlp_best.fit(X_train_small_tfidf_olddata,y_train_small)
mlp_prediction=mlp_best.predict(X_test_small_tfidf_olddata)
print_confusion_matrix(y_test_small,mlp_prediction,'TFIDF: MLP ')


# In[ ]:




