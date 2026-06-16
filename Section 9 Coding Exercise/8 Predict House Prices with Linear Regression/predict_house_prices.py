# Import libraries
import pandas as pd
import numpy as np

## Load data
from sklearn.datasets import fetch_california_housing

## Splits the data set into training and testing sets
from sklearn.model_selection import train_test_split

## Get the model and implemented the linear regression algorithm to predict house prices.
from sklearn.linear_model import LinearRegression

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model.
from sklearn.metrics import mean_squared_error, r2_score

## Loads the California housing data sets and returns it as a pandas data frame.
## This data set contains features like median income, house age, average number of rooms, and others, along with the target variable, which is the median house value, which is actually in hundred of thousands of dollars. If its 1 it means it's 100,000, if it's 1.5, it means 1.5 million.
housing = fetch_california_housing(as_frame=True)

## Create a Dataframe from the dataset
## Convert the data set into a data frame for easy manipulation
df = housing.frame  # type: ignore

## Display the first few rows of the data set just to see how it looks like.
## Print the first five rows of California housing data frame to give us an overview
print("California Housing Data:")
## Print first 5 rows
print(df.head())

## We can now check data overview terminal --> python predict_house_prices.py

## Get the Features (independent variables) and target (dependent variable)
## We need all the columns except the last column, which is the median house value in the features
## Features - this will drop the column median house value
X = df.drop('MedHouseVal', axis=1)


## Target - what do I want to uh the target is target variable is the median house value which represents the median house value in hundreds of thousands of dollars.
y = df['MedHouseVal']

## Split the data set into training and testing
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
## Initializes the linear regression model
model = LinearRegression()

## fit() function trains the model using the training data X_train and y_train
model.fit(X_train, y_train)

## Make predictions as using the trained model to predict the median house value on the test set.
y_pred = model.predict(X_test)

## Evaluate the model based on our training and testing using MSE and R2 score
## Calculates the mean_squared_error, which measures the average squared difference between the actual and the predicted house values.
## r2_score indicates how well the model fits the data. r2_score closer to one means better performance.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

## Print the result
print(f"MeanS quare Error: {mse}")
print(f"R2 Score: {r2}")

## Display the coefficients of the model
print(f"Model Coefficients: ")
## displays the intercept of linear regression model
print(f"Intercept: {model.intercept_}")
## displays coefficients for each feature
print(f"coefficients: {model.coef_}")

## Create a data frame for the coefficients with their corresponding feature names
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['coefficient'])

print(f"coefficients for each feature")
print(coef_df)

# Test the model with new data
## Create a new data object
## 
new_data = pd.DataFrame({
    'MedInc':[5],
    'HouseAge':[30],
    'AveRooms': [6],
    'AveBedrms': [1],
    'Population':[500],
    'AveOccup':[3],
    'Latitude':[34.05],
    'Longitude':[-118.25]
})

## Get the predicted price
predicted_price = model.predict(new_data)
## Print the result
print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")