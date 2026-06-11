# Install required packages       
## pip install xgboost pandas scikit-learn matplotlib seaborn

# import libraries

## These are data manipulation and numerical operations XGBoost.
## Data manipulation library allowing you to load and manipulate data in a structured format like a DataFrame
import pandas as pd
## Matrix operations and mathematical functions
import numpy as np

## Xgboost classifier
import xgboost as xgb

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Labelencoder
from sklearn.preprocessing import LabelEncoder

# Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Used for data visualization
import matplotlib.pyplot as plt

## data visualization - create a heat map from the confusion matrix visualization
import seaborn as sns

# Load the dataset (replace with actual file path)
## If you have a different dataset replace this with your actual file path
df = pd.read_csv('data/HR-Employee-Attrition.csv')

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Preprocess the dataset
# Drop irrelevant columns - removes all the irrelevant columns that don't affect iteration prediction.
df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
## This encodes the categorical features into numerical values using label encoder which is helpful for making sure it doesn't give us error, because the classifier doesn't understand categorical values, it understands numerical values.
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split features and target
## Drop the attrition because that is our feature
X = df.drop('Attrition', axis=1)
## y variable the target variable where 0 is employee stayed and 1 is employee left the company
y = df['Attrition']

# Train-test split
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes (stayed and left) are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the XGBoost classifier
## This initializes the XGBoost classifier and trains it on the training set.
## use_label_encoder=False disables the deprecated label encoder.
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
## fit our model and pass it the data X_train and y_train to train the model.
model.fit(X_train, y_train)

# Make predictions
## This uses the trained model to make predictions on the test set (X_test).
y_pred = model.predict(X_test)

# Evaluate the model
## Calculates the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
## Generates a confusion matrix which we will be displaying using the using matplotlib
conf_matrix = confusion_matrix(y_test, y_pred)
## Provides precision recall F1 score and support in a tabular format
class_report = classification_report(y_test, y_pred)

# This displays all the evaluation metrics for us to see on the command prompt.
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", class_report)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualize the confusion matrix
## So this entire code here will visualize the confusion matrix using Seaborn for better readability than what we have in our text
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()