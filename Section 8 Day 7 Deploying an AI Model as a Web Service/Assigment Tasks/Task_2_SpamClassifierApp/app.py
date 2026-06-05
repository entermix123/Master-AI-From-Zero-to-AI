"""
Task 2 - Spam Classifier Web Service with CSV Bulk Upload
Run with:
    terminal --> python app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import joblib
import csv
import io

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

    text_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(text_tfidf)[0]
    label      = 'SPAM' if prediction == 1 else 'HAM (Not Spam)'

    return render_template('index.html', result=label, input_text=email_text)


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files.get('csv_file')

    if not file or file.filename == '':
        return render_template('index.html', csv_error="No file selected.")

    if not file.filename.endswith('.csv'):
        return render_template('index.html', csv_error="Please upload a .csv file.")

    # Read and decode the uploaded CSV
    stream  = io.StringIO(file.stream.read().decode('utf-8'))
    reader  = csv.reader(stream)
    rows    = list(reader)

    if len(rows) < 2:
        return render_template('index.html', csv_error="CSV is empty or has no data rows.")

    header   = rows[0]
    data_rows = rows[1:]

    # Expect a single column named 'email' (case-insensitive), or just use column 0
    col_index = 0
    if header and header[0].strip().lower() in ('email', 'text', 'message'):
        col_index = 0

    results = []
    for row in data_rows:
        if not row or len(row) <= col_index:
            continue
        text       = row[col_index].strip()
        tfidf      = vectorizer.transform([text])
        prediction = model.predict(tfidf)[0]
        label      = 'SPAM' if prediction == 1 else 'HAM'
        results.append({'text': text, 'label': label})

    return render_template('index.html', csv_results=results)


if __name__ == '__main__':
    app.run(debug=True)
