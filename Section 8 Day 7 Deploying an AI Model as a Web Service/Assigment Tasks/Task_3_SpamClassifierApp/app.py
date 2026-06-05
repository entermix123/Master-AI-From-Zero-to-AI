"""
Task 3 - Spam Classifier with Confidence Scores and Visualizations
Run with:
    terminal --> python app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import joblib
import csv
import io

app = Flask(__name__)

model      = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('spam_vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email_text', '').strip()

    if not email_text:
        return render_template('index.html', error="Please enter some text.")

    tfidf        = vectorizer.transform([email_text])
    prediction   = model.predict(tfidf)[0]
    probabilities = model.predict_proba(tfidf)[0]  # [prob_ham, prob_spam]

    label        = 'SPAM' if prediction == 1 else 'HAM'
    spam_pct     = round(probabilities[1] * 100, 1)
    ham_pct      = round(probabilities[0] * 100, 1)

    return render_template(
        'index.html',
        input_text  = email_text,
        label       = label,
        spam_pct    = spam_pct,
        ham_pct     = ham_pct,
    )


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files.get('csv_file')

    if not file or file.filename == '':
        return render_template('index.html', csv_error="No file selected.")
    if not file.filename.endswith('.csv'):
        return render_template('index.html', csv_error="Please upload a .csv file.")

    stream    = io.StringIO(file.stream.read().decode('utf-8'))
    reader    = csv.reader(stream)
    rows      = list(reader)

    if len(rows) < 2:
        return render_template('index.html', csv_error="CSV is empty or has no data rows.")

    header    = rows[0]
    data_rows = rows[1:]
    col_index = 0
    if header and header[0].strip().lower() in ('email', 'text', 'message'):
        col_index = 0

    results     = []
    spam_count  = 0
    ham_count   = 0

    for row in data_rows:
        if not row or len(row) <= col_index:
            continue
        text          = row[col_index].strip()
        tfidf         = vectorizer.transform([text])
        prediction    = model.predict(tfidf)[0]
        probabilities = model.predict_proba(tfidf)[0]
        label         = 'SPAM' if prediction == 1 else 'HAM'
        spam_pct      = round(probabilities[1] * 100, 1)
        ham_pct       = round(probabilities[0] * 100, 1)

        if label == 'SPAM':
            spam_count += 1
        else:
            ham_count += 1

        results.append({
            'text':     text,
            'label':    label,
            'spam_pct': spam_pct,
            'ham_pct':  ham_pct,
        })

    return render_template(
        'index.html',
        csv_results = results,
        spam_count  = spam_count,
        ham_count   = ham_count,
    )


if __name__ == '__main__':
    app.run(debug=True)