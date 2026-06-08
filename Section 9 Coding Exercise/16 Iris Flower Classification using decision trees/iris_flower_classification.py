# inport libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Load data
from sklearn.datasets import load_iris

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Implements the decision tree classifier from scikit learn
from sklearn.tree import DecisionTreeClassifier

## Import metrics from scikit learn
from sklearn import metrics

## Function to visualize the trained decision tree
from sklearn.tree import plot_tree

## JUsed to visualize the decision tree
import matplotlib.pyplot as plt

# Load the load and display the iris data set
## Load the iris data set which includes features like sepal length, sepal width, petal length and petal width
iris = load_iris()

## Create a DataFrame from the Iris dataset
## Converts the data set into pandas data frame for easier manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

## Add the species column (target)
df['species'] = iris.target

## Display the first few rows of the data set
print("Iris Dataset")
print(df.head())

## split the dataset into features (X) and target (y)
## The features include sepal length, sepal width, petal length and petal width
X = df.drop('species', axis=1)

## Target variable
## A species is either setosa versicolor or virginica
y = df['species']

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 30% of the data to be test, whereas 70% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
## Initialize classifier
classifier = DecisionTreeClassifier()

## Trains the classifier on the training set
classifier.fit(X_train, y_train)

## Make predictions on the test set - Predicts the species of the iris flowers in the test set
y_pred = classifier.predict(X_test)

# Evaluate the model using accuracy, confusion matrix and classification report
## accuracy score calculates the accuracy of the model, that is the percentage of correct predictions
accuracy = metrics.accuracy_score(y_test, y_pred)
## It generates the confusion matrix, which shows the number of correct and incorrect predictions for each class
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
## Classification report provides a detailed classification report including precision recall and F1 score for each class
class_report = metrics.classification_report(y_test, y_pred)

## Print results
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix")
print(conf_matrix)
print("\Classification Report")
print(class_report)

# Visualize the decision Tree
plt.figure(figsize=(12,6))
## So what feature_names=iris.feature_names does is is the labels the features which is the sepal width
## class_names=iris.target_names labels the target classes setosa, versicolor and virginica, and filled true colors the tree nodes based on the class they predict, providing a visual understanding of the decision process
## filled=True - filled true colors the tree nodes based on the class they predict, providing a visual understanding of the decision process
plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
## Set plot title
plt.title("Decision Tree for Iris Flower Classification")
## Print the plot
plt.show()
