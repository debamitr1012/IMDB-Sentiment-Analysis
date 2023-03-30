#!/usr/bin/env python
# coding: utf-8

# ## Learning about data

# In[1]:


import numpy as np 
import pandas as pd
data = pd.read_csv("C:/Users/91983/IMDB Dataset.csv/IMDB Dataset.csv")
data.head()


# In[2]:


data.info()


# __observation(obv)__ 
# 1) There are 50,000 entries in dataset \
# 2) There are two features __review__ and __sentiment__ \
# 3) Both features are __Object__ so they are basically __"text"__ \
# 4) Both features have 50K data in there respective column, so no __null__ value(i.e. there are no empty rows in features)

# In[3]:


data.isnull().sum()


# In[4]:


data.sentiment.value_counts()


# In[5]:


data.review.value_counts().head(2)


# In[6]:


data.duplicated().value_counts()


# ## Data Cleaning

# In[7]:


data = data.sample(30000)


# In[8]:


data.shape, data.info(), data.sentiment.value_counts()


# In[9]:


data.drop_duplicates(inplace=True)


# In[10]:


data.duplicated().value_counts()


# In[13]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
def clean_review(review, stemmer = PorterStemmer(), stop_words = set(stopwords.words("english"))):
    soup = BeautifulSoup(review, "html.parser")
    no_html_review = soup.get_text().lower()
    clean_text = []
    for word in review.split():
        if word not in stop_words and word.isalpha():
            clean_text.append(stemmer.stem(word))
    return " ".join(clean_text)


# In[14]:


data.review = data.review.apply(clean_review)


# In[15]:


data.review.iloc[3537]


# In[16]:


data


# ## Vectorizer

# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)


# In[18]:


X = cv.fit_transform(data.review).toarray()


# In[19]:


X.shape


# In[20]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data.sentiment = lb.fit_transform(data.sentiment)


# In[21]:


y = data.iloc[:,-1].values


# In[22]:


y.shape


# ## Model Building

# In[23]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=data.sentiment)


# In[24]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()
clf1.fit(train_X, train_Y)
clf2.fit(train_X, train_Y)
clf3.fit(train_X, train_Y)


# In[25]:


predict1 = clf1.predict(test_X)
predict2 = clf2.predict(test_X)
predict3 = clf3.predict(test_X)


# In[26]:


from sklearn.metrics import accuracy_score
print("Gaussin NaiveBayes:", accuracy_score(predict1, test_Y))
print("Multinomial NaiveBayes:", accuracy_score(predict2, test_Y))
print("Benouli NaiveBayes:", accuracy_score(predict3, test_Y))


# ## Deployment

# In[27]:


features_dict = {}
for i in range(len(cv.get_feature_names())):
    features_dict[cv.get_feature_names()[i]] = i


# In[28]:


import pickle


# In[29]:


pickle.dump(data, open("dataframe.pkl", "wb"))


# In[30]:


pickle.dump(features_dict, open("features_dict.pkl", "wb"))

