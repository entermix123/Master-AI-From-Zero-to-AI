# import libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Converts text data into numerical features using term. In the CSV data file we text information. We want to convert/vectorize it.
from sklearn.feature_extraction.text import TfidfVectorizer

## Naive Bayes classifier suitable for text data
from sklearn.naive_bayes import MultinomialNB

## Metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
from sklearn.metrics import  accuracy_score, classification_report

# Load the dataset
## https://www.kaggle.com/datasets/algord/fake-news
## If you have different dataset, replace the FakeNewsNet.csv with the actual one.
## The dataset should have a text column containing 'news' and 'content' and a label column which with values like 'fake' or 'real' or similar to that we have. Or you can change it accordingly.
df = pd.read_csv('data/FakeNewsNet.csv')

# Display the first few rows of the dataset
print("Dataset Sample:\n", df.head())

# Define features (X) and target (y)
X = df['title']   # Assuming the dataset has a 'text' column for the news content
y = df['real']    # Assuming the dataset has a 'label' column with values 'fake' or 'real'

# Split the dataset into training and testing sets with stratification
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes are proportionally represented in the training and test sets whether it's fake or real news
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the TfidfVectorizer to convert text into numerical features
## Converts text data into TF-IDF scores. The stop_words='english' remote's common English stopwords to focus on important words we don't want 'and', 'the', 'if', all those words
## max_df=0.7 ignores words appearing in more than 70% of document, as they are likely to common to be useful. For example, if there are words like 'news' or 'channel' we don't need them because they will be everywhere.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
## fit_transform(X_train) learns the vocabulary from X_train data and transforms it into TF-IDF vectors
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform the training data
## vectorizer.transform(X_test)  converts X_test data into tf TF-IDF vectors using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)        # Only transform the test data

# Initialize the Naive Bayes classifier
## Initializes the Naive Bayes classifier which is effective for text classification task where features like words are multinomial distributed.
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
## This uses the trained model to classify the articles in the test set X_test_tfidf. The predicted labels y_pred indicate whether each article is fake or real.
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
## This calculates the accuracy of the model, that is, the percentage of correctly classified articles in the test set.
accuracy = accuracy_score(y_test, y_pred)
## Generates a detailed report including precision recall and F1 score for each class whether it is fake or real. 
## zero_division=1 parameter avoids warnings for any labels without true samples by setting the undefined metrics to 1.0.
report = classification_report(y_test, y_pred, zero_division=1)

## Print the accuracy and the classification report. So this will give us a good idea about how good the model is.
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)
