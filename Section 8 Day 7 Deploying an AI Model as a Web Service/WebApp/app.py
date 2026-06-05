from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('imdb_sentiment_model.pkl')
vectorizer = joblib.load('imdb_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    review = request.form['review']

    if not review:
        return "No Review Avalable"

    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]

    sentiment = 'posistive' if prediction == 1 else 'negative'

    return f"The movie review is <b>{sentiment}</b>"

if __name__ == "__main__":
    app.run(debug=True)