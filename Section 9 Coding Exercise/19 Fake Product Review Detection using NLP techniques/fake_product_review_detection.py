# import libraries
## handling the data set
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd 

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Converts text into numerical features by calculating term frequency. Inverse document frequency.
from sklearn.feature_extraction.text import TfidfVectorizer

## Simple classifier to predict whether a review is fake or genuine
from sklearn.linear_model import LogisticRegression

## Metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
from sklearn.metrics import  accuracy_score, classification_report

# Load the dataset (replace 'review.csv' with the actual dataset path)
df = pd.read_csv('data/fake_reviews_dataset.csv')
df = df.rename(columns={'text_': 'review_text'})
df['label'] = df['label'].map({'OR': 'genuine', 'CG': 'fake'})

# Display the first few rolls of the dataset
print("Dataset Seample\n", df.head())

# Define features (X) and target (y)
X = df['review_text']   # Assuming the dataset has a 'revirew_text' column
y = df['label']     # Assuming the dataset has 'label' column with values 'fake' or 'genuine'

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Tfidfvectorizer
## stop_words='english' removes common English stopwords to focus on important words, it will remove words like 'and', 'if', 'that', 'the' 'all those' and easy words which are used quite a lot
## max_df=0.7 ignores words that appear in more than 70% of the documents. If there is something that comes quite a lot like food, food, food, since it's a review, it might have that review text over there.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
## This learns vocabulary from Xtrain and transforms it into TF-IDF vector
X_train_tfidf = vectorizer.fit_transform(X_train)   # Fit and transform the training
## This transforms X test to TF-IDF vectors using the vocabulary from Xtrain
X_test_tfidf = vectorizer.transform(X_test)         # Only transfor the test data

# Initialize the Logistic Regression Classifier
## This logistic regression initializes the logistic regression classifier suitable for binary classification task in our case
model = LogisticRegression()

# Train the model
## This trains the logistic regression model on the training data. Learning the relationship between TF-IDF vectors and labels.
model.fit(X_train_tfidf, y_train)

# Make the predictions on the test set
## It uses the trained model to predict labels for test data. Identifying whether reviews are fake or genuine.
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
## Calculates the overall accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
## Generates a detailed report with metrics like precision, recall and F1 score for each class. Fake and genuine.
report = classification_report(y_test, y_pred)

# Print results
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report\n", report)