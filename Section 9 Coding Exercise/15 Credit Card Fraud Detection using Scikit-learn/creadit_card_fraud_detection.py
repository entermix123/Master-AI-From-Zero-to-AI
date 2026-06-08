# inport libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## 
from sklearn.preprocessing import StandardScaler


## Robust classifier that works well for binary classification tasks
from sklearn.ensemble import RandomForestClassifier

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## classification_report 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('data/creditcard.csv')

# Display first few rows from dataset sample
print("Dataset Sample:\n", df.head())

# Separated features X and target Y
X = df.drop('Class', axis=1)        # 'Class' column is the target, with 0 for non-fraud
y = df['Class']

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features for numerical stability
## StandardScaler() - standardizes features to improve numerical stability
scaler = StandardScaler()

## The fit transform fits the scalar on x train and transforms it
X_train = scaler.fit_transform(X_train)

## Transforms the x underscore test using the scaling parameters from x underscore train
X_test = scaler.transform(X_test)

# Initialize and train the model 
## Initialize the model with 100 trees
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
model = RandomForestClassifier(n_estimators=100, random_state=42)

## Train the model on the training data that we have
model.fit(X_train, y_train)

## Make prediction on the test set
## this function uses the trained model model to predict labels for X_test.
y_pred = model.predict(X_test)

# Evaluate the model
## This calculates the overall accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
## This provides precision recall and F1 score for each class fraud and non fraud
report = classification_report(y_test, y_pred)
## Generates a confusion matrix showing true positives, false positives, true negatives and false negatives. So We'll get an matrix of these four elements.
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)
