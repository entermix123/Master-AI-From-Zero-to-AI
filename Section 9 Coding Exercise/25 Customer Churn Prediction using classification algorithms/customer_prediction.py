# import libraries

## pandas used for loading and manipulating the data set
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## This scales features to have a mean of zero and standard deviation of one helping improve model stability.
from sklearn.preprocessing import StandardScaler

## Binary classification problems in which in our case we are going to predict churn or no churn
from sklearn.linear_model import LogisticRegression

## Metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
from sklearn.metrics import  accuracy_score, classification_report

# Load the dataset
## replace the customer churn to any other data that you should work with. The data should have data set should have various features related to customer information and the churn column indicating churn status. You need to have that in your data set.
df = pd.read_csv('data/customer_churn.csv')

# Display the first few rows of the dataset
## This displays the first few rows of the data set to confirm it's loaded correctly
print("Dataset Sample:\n", df.head())

# Basic data preprocessing: handle missing values (if any)
## Removes any rows with missing values to ensure data integrity
df = df.dropna()

# Convert categorical columns to numeric using one-hot encoding
## If it's not numerical it will convert into numerical.
df_encoded = pd.get_dummies(df, columns=['gender', 'contract_type', 'payment_method'])

# Define features (X) and target (y)
## Drop the churn in along axis one. So that will get us the features.
X = df_encoded.drop('churn', axis=1)  # Features
## Use churn as target variable
y = df_encoded['churn']               # Target variable

# Split the dataset into training and testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes (churn and non churn) are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling for numerical stability
## Standardizes features to have a mean of zero and a standard deviation of one.
scaler = StandardScaler()
## Calculate scaling parameters Metres mean and standard deviation on the training set and applies proper scaling.
X_train = scaler.fit_transform(X_train)
## Appliy the same scaling parameters from X_train to the test set (X_test) for consistency.
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression classifier
## This initializes the logistic regression classifier, which is commonly used for binary classification tasks like churn prediction.
model = LogisticRegression()

# Train the model
## The fit function trains the logistic regression model on the training data of X_train and y_train, learning the relationship between the features and the churn status.
model.fit(X_train, y_train)

# Make predictions on the test set
## This uses the trained model to predict the churn status for customers in the test set which is X_test, the predictions y_pred indicate whether each customer is classified as having churn or not.
y_pred = model.predict(X_test)

# Evaluate the model
## This calculates the accuracy of the model, which is the percentage of correctly classified customers in the test set
accuracy = accuracy_score(y_test, y_pred)
## Generates a report including precision recall and F1 score for each class.
report = classification_report(y_test, y_pred, zero_division=1)

# Print results
## So this will print out accuracy, which will show the overall percentage of correctly classified samples.
print(f"\nAccuracy: {accuracy}")
## And the classification report will show us the metrics like precision recall and F1 score for each class.
print("\nClassification Report:\n", report)
