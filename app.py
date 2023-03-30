import streamlit as st
import pickle
import sklearn
from nltk.stem.porter import PorterStemmer
import numpy as np
data = pickle.load(open("dataframe.pkl", "rb"))
features_dict = pickle.load(open("features_dict.pkl", "rb"))
def vectorize(dataframe):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    cv = CountVectorizer(max_features=5000)
    X = cv.fit_transform(dataframe.review).toarray()
    data.sentiment = lb.fit_transform(dataframe.sentiment)
    y = data.iloc[:, -1].values
    return X, y
def stemming(review, stemmer=PorterStemmer()):
    stem_word = []
    for i in review.split():
        stem_word.append(stemmer.stem(i))
    return stem_word
def make_vector(value):
    input_vector = np.zeros(5000)
    for i in range(len(value)):
        if value[i] in features_dict:
            input_vector[features_dict[value[i]]] = 1 + input_vector[features_dict[value[i]]]
    input_vector = input_vector.reshape(1, -1)
    return input_vector
def naive_bayes_model(x, y, input_vector):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2, random_state=42, stratify=data.sentiment)
    clf = MultinomialNB()
    clf.fit(train_X, train_Y)
    predict = clf.predict(input_vector)
    return predict
st.title("Sentiment analysis")
container = st.container()
selected_text = container.text_input("What's on you mind?")
container.write(selected_text)
a = stemming(selected_text)
a = make_vector(a)
X, y = vectorize(data)
model = naive_bayes_model(X, y, a)
if model[0] == 1:
    container.write(f"You are Positive")
    container.write("( ͡° ͜ʖ ͡°)")
elif model[0] == 0:
    container.write(f"You are negative")
    container.write("¯\_(ツ)_/¯")