"""
Task 1 - Spam Classifier Web Service
Run with:
    terminal --> python app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer (run train_model.py first)
model      = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('spam_vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email_text', '').strip()

    if not email_text:
        return render_template('index.html', result="Please enter some text.", label="")

    # Vectorize and predict
    text_tfidf  = vectorizer.transform([email_text])
    prediction  = model.predict(text_tfidf)[0]

    label = 'SPAM' if prediction == 1 else 'HAM (Not Spam)'

    return render_template('index.html', result=label, input_text=email_text)


if __name__ == '__main__':
    app.run(debug=True)
