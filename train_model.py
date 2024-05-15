import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from preprocessing import clean_text, preprocess_data  

def train_model(data):
    data = preprocess_data(data)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  
    vectors = tfidf.fit_transform(data['Sentence']).toarray()

    X_train, X_test, y_train, y_test = train_test_split(vectors, data['Sentiment'], test_size=0.2, random_state=42)
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)

    y_pred = mnb.predict(X_test)
    print(classification_report(y_test, y_pred))  

    return mnb, tfidf

label_map = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

def get_predict(text, model, tfidf):
    cleaned_text = clean_text(text)  
    vector_text = tfidf.transform([cleaned_text])  
    prediction = model.predict(vector_text)[0]
    text_label = label_map.get(prediction, 'Unknown')  
    return text_label