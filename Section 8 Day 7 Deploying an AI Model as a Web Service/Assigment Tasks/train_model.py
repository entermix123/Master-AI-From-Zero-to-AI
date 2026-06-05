"""
Task 1 - Train a Spam Email Classifier (Naive Bayes)
Run this once to generate the saved model and vectorizer:
    terminal --> python train_model.py
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── Load a binary subset of 20 Newsgroups as a stand-in spam/ham dataset ──
# 'talk.politics.misc' acts as "spam", 'rec.sport.baseball' acts as "ham"
categories = ['talk.politics.misc', 'rec.sport.baseball']
label_names = {0: 'ham (not spam)', 1: 'spam'}

train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data  = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers', 'footers', 'quotes'))

X_train, y_train = train_data.data, train_data.target
X_test,  y_test  = test_data.data,  test_data.target

# ── Vectorize with TF-IDF ──
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ── Train Naive Bayes ──
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ── Evaluate ──
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# ── Save model and vectorizer ──
joblib.dump(model,      'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'spam_vectorizer.pkl')
print("Model saved: spam_classifier_model.pkl")
print("Vectorizer saved: spam_vectorizer.pkl")
