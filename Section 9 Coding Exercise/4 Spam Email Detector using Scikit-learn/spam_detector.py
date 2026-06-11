# import libraries
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Binary classification problems in which in our case is spam versus not spam
from sklearn.linear_model import LogisticRegression

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## Precision score, which measures the proportion of positive predictions that are actually correct
## Recall score, which measures the proportion of actual positives that are correctly identified
## F1 score which is the harmonic mean of precision and recall
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

## data visualization - It is built on top of matplotlib and makes it easy to create beautiful and informative statistical graphics
import seaborn as sns

## allows us to plot graphs and charts such as the confusion matrix that we will do
import matplotlib.pyplot as plt

# Loading Dataset
## Split the dataset to training and testing datasets
data = pd.read_csv('spam.csv')

## drop the column named spam from the data set to create the feature matrix X
## X will contain all the columns except the target column spam
## This will be the data used to predict whether an email is spam or not
X = data.drop('spam', axis=1)

## y is the target variable, which is the spam column, which contains the binary labels zero for not spam and one for spam
y = data['spam']

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our our target which is spam
## Test size is out of all the data that I have, I want 20% of the data to be test, whereas 80% of the data to be used as a training data
## Setting a random state ensures reproducibility
## This will give me variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the logistic regression model to classify emails as spam or not spam
## initialize logistic regression model
model = LogisticRegression()

## trains the logistic regression model using the training features which is X_train and the training labels y_train
model.fit(X_train, y_train)

## making predictions - uses the model to predict the labels spam or not spam nn the test set X_test, the predicted labels are stored in the variable y_pred
y_pred = model.predict(X_test)

## Optional - Print data
print(X_test)

## Evaluate the model using the accuracy, confusion matrix, precision, recall and F1 score
## calculating the accuracy of the model, which is the percentage of correct predictions both spam and not spam
accuracy = accuracy_score(y_test, y_pred)

## calculates precision, which is the proportion of emails predicted as spam that are actually spam, which is that is like true positives divided by true positives plus false positives
precision = precision_score(y_test, y_pred)

## calculates the recall which is the proportion of actual spam emails that are correctly predicted - true positives divided by true positives plus false negatives.
recall = recall_score(y_test, y_pred)

## harmonic mean of precision and recall. It provides a single metric that balances both precision and recall
f1 = f1_score(y_test, y_pred)

## Print variables
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")


# Visualize the confusion matrix using Seaborn heatmap
## generates the confusion matrix a table that shows it will show me four different sections
## One is true positive which is emails that were correctly predicted as spam. 
## Two is true negatives which are emails that were correctly predicted as not spam. 
## Tree is false positives - emails that were incorrectly predicted spam but were not spam. 
## Four - false negatives which are emails that were incorrectly predicted as not spam but were actually spam.
cm = confusion_matrix(y_test, y_pred)

## visualize the confusion matrix as heatmap 
## annot=True - annotates the heatmap with the actual values from the confusion matrix
## fmt='d' - ensures that the values are shown as integers and not scientific notation
sns.heatmap(cm, annot=True, fmt='d')

# Set title and print the heatmap
plt.title('Confusion Matrix')
plt.show()