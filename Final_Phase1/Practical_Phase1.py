#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Practical Phase1 
## Amir Pourmand
## Stu No: 99210259


# # Download Required Packages

# In[ ]:


get_ipython().system('gdown --id 15JJ6ZysFM57tlUjXo2nHVhkGwePbVMVV')


# In[ ]:


get_ipython().system('pip install contractions')
get_ipython().system('pip install unidecode')
get_ipython().system('pip install word2number')


# ## Imports

# In[ ]:


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


# ## Preprocessing Text Functions

# In[ ]:


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


# # Split the data

# In[ ]:


raw_data = pd.read_csv('dataset.csv')
raw_data['sentiment'] = raw_data['sentiment'].apply(lambda input: 1 if input == 'positive' else 0)

#0 means unprocessed
X_train_0,X_test_0,y_train,y_test=train_test_split(raw_data['comment'],raw_data['sentiment'],test_size=0.2)

# 1 means one level of pre-processing
X_train_1,X_test_1=X_train_0.copy(),X_test_0.copy()

# 2 means two level of pre-processing
X_train_2,X_test_2=X_train_0.copy(),X_test_0.copy()


# ## Preprocess Data

# In[ ]:


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


# In[ ]:


X_train_1 = process_level1(X_train_1)
X_test_1 = process_level1(X_test_1)

X_train_2 = process_level2(X_train_2)
X_test_2 = process_level2(X_test_2)


# ## Bag of Words Representation

# In[ ]:


def convert_to_BOW(train,test):
    vectorizer = CountVectorizer(max_df=0.4,min_df=0.01,lowercase=False)
    X_train_transformed = vectorizer.fit_transform(train)
    X_test_transformed = vectorizer.transform(test)
    return X_train_transformed,X_test_transformed


# In[ ]:


X_train_0_BOW,X_test_0_BOW = convert_to_BOW(X_train_0,X_test_0)
X_train_1_BOW,X_test_1_BOW = convert_to_BOW(X_train_1,X_test_1)
X_train_2_BOW,X_test_2_BOW = convert_to_BOW(X_train_2,X_test_2)


# In[ ]:


X_train_0_BOW


# In[ ]:





# # Models 

# ## 1- Regression Model

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[ ]:


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


# In[ ]:


from sklearn.linear_model import LogisticRegression
  
def regression(X_train,X_test,y_train,**kwarg):
    clf = LogisticRegression(**kwarg).fit(X_train, y_train)
    return clf.predict(X_test),clf


# In[ ]:


print_confusion_matrix(y_test,regression(X_train_0_BOW,X_test_0_BOW,y_train)[0],'Regression not preprocessed')
print_confusion_matrix(y_test,regression(X_train_1_BOW,X_test_1_BOW,y_train)[0],'Regression preprocessed level1')
print_confusion_matrix(y_test,regression(X_train_2_BOW,X_test_2_BOW,y_train)[0],'Regression preprocessed level2')


# ## 2- KNN Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
def knn(X_train,X_test,y_train):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    return neigh.predict(X_test),neigh


# In[ ]:


print_confusion_matrix(y_test,knn(X_train_0_BOW,X_test_0_BOW,y_train)[0],'KNN Not Preprocessed')
print_confusion_matrix(y_test,knn(X_train_1_BOW,X_test_1_BOW,y_train)[0],'KNN Preprocessed Level1')
print_confusion_matrix(y_test,knn(X_train_2_BOW,X_test_2_BOW,y_train)[0],'KNN Preprocessed Level2')


# ## 3- SVM Model

# In[ ]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
def svm(X_train,X_test,y_train):
    clf = make_pipeline( SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    return clf.predict(X_test),clf


# In[ ]:


print_confusion_matrix(y_test,svm(X_train_0_BOW,X_test_0_BOW,y_train)[0],'SVM Not Preprocessed')
print_confusion_matrix(y_test,svm(X_train_1_BOW,X_test_1_BOW,y_train)[0],'SVM proprocessed level1')
print_confusion_matrix(y_test,svm(X_train_2_BOW,X_test_2_BOW,y_train)[0],'SVM preprocessed level2')


# # Word2Vector and HyperParameter Effect

# In[ ]:


# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
# https://www.kaggle.com/kstathou/word-embeddings-logistic-regression


# In[ ]:


import gensim.models

w2v = gensim.models.Word2Vec( [row.split() for row in X_train_2], 
                             min_count=50,
                            window=10, 
                            size=300)


# In[ ]:


w2v.most_similar('king')


# In[ ]:


def document_vector(doc):
    doc = [word for word in doc.split() if word in w2v.wv.vocab]
    return np.mean(w2v[doc], axis=0)


# In[ ]:


X_train_w2v = X_train_2.apply(document_vector)
X_test_w2v = X_test_2.apply(document_vector)


# 

# ## Logistic Regression

# In[ ]:


lr_w2v_pred,regression_w2v = regression(list(X_train_w2v),list(X_test_w2v),y_train,
                                            C=150,max_iter=10000)
print_confusion_matrix(y_test,lr_w2v_pred,'Word2Vector: Regression preprocessed level2')

lr_bow_pred,regression_BOW = regression(X_train_2_BOW,X_test_2_BOW,y_train)
print_confusion_matrix(y_test,lr_bow_pred,'BagOfWords: Regression preprocessed level2')


# ## KNN

# In[ ]:


knn_w2v_pred,knn_w2v = knn(list(X_train_w2v),list(X_test_w2v),y_train)

print_confusion_matrix(y_test,knn_w2v_pred,'Word2vec: Preprocessed Level2')

knn_bow_pred,knn_bow = knn(X_train_2_BOW,X_test_2_BOW,y_train)
print_confusion_matrix(y_test,knn_bow_pred,'BagOfWords: KNN Preprocessed Level2')


# ## SVM

# In[ ]:


svm_w2v_pred,svm_w2v=svm(list(X_train_w2v),list(X_test_w2v),y_train)
svm_bow_pred,svm_bow = svm(X_train_2_BOW,X_test_2_BOW,y_train)

print_confusion_matrix(y_test,svm_w2v_pred,'W2V: SVM preprocessed level2')
print_confusion_matrix(y_test,svm_bow_pred,'BOW: SVM preprocessed level2')


# ## MLP

# ### TF-IDF Vectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer( min_df=0.01,max_df=0.5)
X_train_2_tfidf=vectorizer.fit_transform(X_train_2)
X_test_2_tfidf = vectorizer.transform(X_test_2)


# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

grid = {
    'hidden_layer_sizes':[(80),(70,),(40,10),(90)],
}
mlp = MLPClassifier(learning_rate='adaptive',solver='adam')
mlp_cv = GridSearchCV (estimator=mlp,param_grid=grid,cv=2)

mlp_cv.fit(X_train_2_tfidf,y_train)

mlp_prediction=mlp_cv.predict(X_test_2_tfidf)
print_confusion_matrix(y_test,mlp_prediction,'TFIDF: MLP ')


# In[ ]:


pd.DataFrame(mlp_cv.cv_results_)


# In[ ]:


mlp_90 = MLPClassifier(learning_rate='adaptive',solver='adam',hidden_layer_sizes=(90))
mlp_90.fit(X_train_2_tfidf,y_train)
mlp_prediction=mlp_90.predict(X_test_2_tfidf)
print_confusion_matrix(y_test,mlp_prediction,'TFIDF: MLP ')


# # Google Drive Save

# In[ ]:


# Load the Drive helper and mount
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import scipy.sparse
import numpy
scipy.sparse.save_npz('/content/drive/MyDrive/DataForColob/ML_Project/X_train_2_BOW',X_train_2_BOW)
scipy.sparse.save_npz('/content/drive/MyDrive/DataForColob/ML_Project/X_test_2_BOW',X_test_2_BOW)

X_train_w2v.to_pickle('/content/drive/MyDrive/DataForColob/ML_Project/X_train_w2v.pkl')
X_test_w2v.to_pickle('/content/drive/MyDrive/DataForColob/ML_Project/X_test_w2v.pkl')

numpy.save("/content/drive/MyDrive/DataForColob/ML_Project/y_train", y_train)
numpy.save("/content/drive/MyDrive/DataForColob/ML_Project/y_test", y_test)


# In[ ]:


import pickle
pickle.dump(svm_w2v, open('/content/drive/MyDrive/DataForColob/ML_Project/SVM.pkl', 'wb'))
pickle.dump(knn_w2v,open('/content/drive/MyDrive/DataForColob/ML_Project/KNN.pkl', 'wb'))
pickle.dump(regression_w2v,open('/content/drive/MyDrive/DataForColob/ML_Project/LR.pkl', 'wb'))


# In[ ]:


import pickle
pickle.dump(mlp_90,open('/content/drive/MyDrive/DataForColob/ML_Project/best.pkl', 'wb'))
pickle.dump(vectorizer,open('/content/drive/MyDrive/DataForColob/ML_Project/vectorizer.pkl', 'wb'))


# In[ ]:




