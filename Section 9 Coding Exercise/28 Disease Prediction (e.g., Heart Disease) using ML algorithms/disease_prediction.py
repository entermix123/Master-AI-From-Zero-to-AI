# import libraries

## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## matrix operation and mathematical function
import numpy as np


## Splits data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV

## This scales features to have a mean of zero and standard deviation of one helping improve model stability.
from sklearn.preprocessing import StandardScaler

## Binary classification problems in which in our case is heart disease or no heart disease
from sklearn.linear_model import LogisticRegression

## random forest classifier - ensemble method for classification
from sklearn.ensemble import RandomForestClassifier

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification - displays true false positives and negatives.
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class - detailed performance report of a model.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Allows us to plot graphs and charts such as the confusion matrix that we will do
import matplotlib.pyplot as plt

## Data visualization - enhances visualization with statistical plots
import seaborn as sns

## We will not see the warnings that are unnecessary for this project.
## If you want to see the warnings you warnings remove this part
import warnings
warnings.filterwarnings('ignore')   


# Step 1: Load Dataset
## If you have a URL or if you're using some other file, make sure that you change that to that particular URL or file
## Data set purpose here is the heart disease data set for predicting heart disease presence
df = pd.read_csv('data/heart.csv')

# Display first few rows
print("Dataset Sample:")
print(df.head())

# Step 2: Data Preprocessing
# Handle missing values (if any)
## df.isnull() identifies missing or null values and .sum() counts the null values in each column
print("\nMissing Values:\n", df.isnull().sum())

# Feature Scaling
## Initializes a scalar
scaler = StandardScaler()
## fit_transform() fits the scaler and scales numerical features
## df.drop('target', axis=1) drops the target column to keep only the feature columns
## We have the last column as target, which we want to drop and just keep the feature columns in this case
scaled_features = scaler.fit_transform(df.drop('target', axis=1))
## X stores scaled features as a DataFrame
X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
## Y stores the target Get variable which is the target column. So Y is our target column.That's what we want to predict. And that's why we keep it in the Y variable.
y = df['target']

# Step 3: Split the Dataset
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes (disease and not disease) are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train Multiple ML Models
# 1. Logistic Regression
## Initialize logistic regression model
log_model = LogisticRegression()
## Trains the model using the training data
log_model.fit(X_train, y_train)
## Predicts outcomes on the test data
log_preds = log_model.predict(X_test)
## Calculate the accuracy of the logistic model
log_accuracy = accuracy_score(y_test, log_preds)
## Print the logistic regression accuracy
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

# 2. Random Forest Classifier
## Initialize random forest model with 100 decision trees
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
## Trains the model using the training data
rf_model.fit(X_train, y_train)
## This predicts outcome on test data
rf_preds = rf_model.predict(X_test)
## Calculate the accuracy of the random forest model
rf_accuracy = accuracy_score(y_test, rf_preds)
## Print the random forest model accuracy
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Step 5: Evaluate the Best Model of the two
## This is where we do the do the model selection, we choose the model with the higher accuracy
## Find best model
best_model = rf_model if rf_accuracy > log_accuracy else log_model
## find best prediction
best_preds = rf_preds if rf_accuracy > log_accuracy else log_preds

## Print hte best results from both models
print("\nBest Model Metrics:")
print("Accuracy Score:", accuracy_score(y_test, best_preds))
## Classification report again will display precision recall F1 score and support
print("Classification Report:\n", classification_report(y_test, best_preds))
## Confusion matrix displays the confusion matrix values which are true negatives true positives false negatives false positives
print("Confusion Matrix:\n", confusion_matrix(y_test, best_preds))

# Step 6: Visualize Confusion Matrix
## Set the plot size
plt.figure(figsize=(8, 6))
## This will display the confusion matrix as a heat map
## annot=True annotates matrix with the values
## cmap='Blues' - blue colors
sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, cmap='Blues', fmt='d')
## Create the title
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
## This will show us the confusion matrix
plt.show()

# Step 7: Make Predictions on New Data
new_data = pd.DataFrame({
    'age': [45],
    'sex': [1],
    'cp': [2],
    'trestbps': [130],
    'chol': [230],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [0.5],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})

# Scale new data
## This applies the previously fitted scalar to the new data
new_data_scaled = scaler.transform(new_data)
## Predict the outcome for the new data
prediction = best_model.predict(new_data_scaled)
## print out prediction for new data at risk disease if prediction = 0 - heart disease, if prediction = 1 - no heart disease
print("\nPrediction for New Data:", "At Risk of Hearth Disease" if prediction[0] == 1 else "No Hearth Disease")
