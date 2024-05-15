from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from train_model import train_model, get_predict
from preprocessing import clean_text

app = Flask(__name__)

data = pd.read_csv('data.csv')
model, tfidf = train_model(data)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = get_predict(text, model, tfidf)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
