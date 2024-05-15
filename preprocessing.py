import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

stopwords_set = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)

    contraction_dict = {
        "isn't": "is not",
        "he's": "he is",
        "wasn't": "was not",
        "there's": "there is",
        "couldn't": "could not",
        "won't": "will not",
        "they're": "they are",
        "she's": "she is",
        "wouldn't": "would not",
        "haven't": "have not",
        "that's": "that is",
        "you've": "you have",
        "he’s": "he is",
        "what's": "what is",
        "weren't": "were not",
        "we're": "we are",
        "hasn't": "has not",
        "you'd": "you would",
        "shouldn't": "should not",
        "let's": "let us",
        "they've": "they have",
        "i'm": "i am",
        "we've": "we have",
        "it's": "it is",
        "don't": "do not",
        "that´s": "that is",
        "I´m": "I am",
        "it’s": "it is",
        "she´s": "she is",
        "he’s": "he is",
        "i’m": "i am",
        "i’d": "i did",
        "he’s": "he is",
        "there’s": "there is"
    }
    text = ' '.join([contraction_dict.get(word, word) for word in text.split()])

    return text

def clean(text):
    words = word_tokenize(text)
    transformed_text = [ps.stem(w.lower()) for w in words if w.isalnum() and ps.stem(w.lower()) not in stopwords_set]
    return ' '.join(transformed_text)

def preprocess_data(data):

    data['cleaned_text'] = data['Sentence'].apply(clean_text)
    data['cleaned_text'] = data['cleaned_text'].astype(str)
    data['cleaned_text'] = data['cleaned_text'].apply(clean)

    le = LabelEncoder()
    data['Sentiment'] = le.fit_transform(data['Sentiment'])
    
    return data

