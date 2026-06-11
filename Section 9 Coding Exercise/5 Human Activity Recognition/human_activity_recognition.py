# import libraries
## pandas used for loading and manipulating the data set
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## random forest classifier - used for human activity recognization
from sklearn.ensemble import RandomForestClassifier

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## Precision score, which measures the proportion of positive predictions that are actually correct
## Recall score, which measures the proportion of actual positives that are correctly identified
## F1 score which is the harmonic mean of precision and recall
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

## data visualization - create a heat map from the confusion matrix visualization
import seaborn as sns

## plotting the confusion matrix
import matplotlib.pyplot as plt

# Load dataset - https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
## loads the Human Activity Recognition which has HR data set
## The data set contains sensor data from accelerometer and gyroscope, and corresponding activity labels will have like walking, sitting, standing, all that stuff inside this data set
df = pd.read_csv('har_data.csv')

## Pre-process the dataset
## Drop the column that I don't need - removes the activity column from the data set leaving only the features which are the sensor readings
X = df.drop('Activity', axis=1)

## target variable contains the activity labels - example walking, sitting and standing.
y = df['Activity']

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target 
## Test size is out of all the data that I have, I want 20% of the data to be test, whereas 80% of the data to be used as a training data
## Setting a random state ensures reproducibility
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Train the RandomForestClassifier
## The random forest is an ensemble method that uses multiple decision trees for classification. That's why we have provided the 100 value to it.
model = RandomForestClassifier(n_estimators=100, random_state=42)

## Train the model
model.fit(X_train, y_train)

## Make predictions on the test set
y_pred = model.predict(X_test)

## Evaluate the model using accuracy, precision, recall, f1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

## Print variables
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_Score: {f1 * 100:.2f}%")

## Visualize the confusion matrix using Seaborn's heatmap
## generates the confusion matrix a table that shows it will show me four different sections
## One is true positive 
## Two is true negatives 
## Tree is false positives
## Four - false negatives
cm = confusion_matrix(y_test, y_pred)

## visualize the confusion matrix as heatmap 
## annot=True - annotates the heatmap with the actual values from the confusion matrix
## fmt='d' - ensures that the values are shown as integers and not scientific notation
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set title and print the heatmap
plt.title('Confusion Matrix')
plt.show()