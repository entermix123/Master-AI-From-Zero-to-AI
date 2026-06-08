# import libraries

## Pandas is used for data manipulation and handling time series data.
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Binary classification problems in which in our case is spam versus not spam
from sklearn.linear_model import LogisticRegression

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## classification_report which
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## data visualization - create a heat map from the confusion matrix visualization
import seaborn as sns

## Plotting the actual predicted stock prices
import matplotlib.pyplot as plt

# Load the Pima Indians Diabetes data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPredigreeFunction', 'Age', 'Outcome']

## Load the dataset
df = pd.read_csv(url, names=column_names)

## Display first few rows of the dataset to see data structure
print("Diabetes Dataset:")
print(df.head())

## Check the dataset structure terminal --> python diabetes_prediction.py

# Get the features. We want all the columns except for outcomes and features
## Features
X = df.drop('Outcome', axis=1)
## Target
y = df['Outcome']

## Splits data into training and testing sets
## Split the data set using the train test split, which splits the data set into 80% for training and 20% for testing, and also will give a random state for it, which is one second
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
## Get hte model
model = LogisticRegression(max_iter=1000)

## Train the model
model.fit(X_train, y_train)

## Make prediction
y_pred = model.predict(X_test)

## Evaluate the model based on this prediction using accuracy, confusion matrix and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

## Check the results of the evaluation
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: {conf_matrix}")
print(f"Classification Report: {class_report}")

## Visualize the confusion matrix I can use the seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

## Test the model with the new data set
new_data = pd.DataFrame({
    'Pregnancies': [5],
    'Glucose': [120],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [80],
    'BMI': [32.0],
    'DiabetesPredigreeFunction': [0.5],
    'Age': [42]
})

## Predicted outcome
predicted_outcome = model.predict(new_data)

## Print the result of the new data
print(f"Predicted Outcome: {'Diabetic' if predicted_outcome[0] == 1 else 'Non-Diabetic'}")
