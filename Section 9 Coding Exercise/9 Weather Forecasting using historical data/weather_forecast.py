# import libraries

## handling the data set
import pandas as pd 

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Get the model and implemented the linear regression algorithm to predict house prices.
from sklearn.linear_model import LinearRegression

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model.
from sklearn.metrics import mean_squared_error, r2_score

## visualize the actual and predicted temperature
import matplotlib.pyplot as plt

## get sample historical dataset
## We can use different dataset from weatherapi
data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Temperature': [30, 32, 34, 31, 29, 28, 35, 33, 30, 31],
    'Humidity': [60, 62, 64, 58, 55, 57, 65, 63, 59, 61],
    'Wind Speed': [10, 12, 8, 11, 9, 10, 13, 12, 10, 11],
    'Precipitation': [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    'Next Day Temperature': [32, 34, 31, 29, 28, 35, 33, 30, 31, 32]
}

## Converting the dictionary into a pandas data frame, calling the data frame function inside pandas
df = pd.DataFrame(data)

## Create feature and target variables
## Create the features - x is the independent variable which are called features that include temperature, humidity, wind speed and precipitation for a particular day
X = df[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation']]
## Create the target - Y is the dependent variable which is the next day's temperature
y = df['Next Day Temperature']

## Split the data set using the train test split, which splits the data set into 80% for training and 20% for testing, and also will give a random state for it, which is one second
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Initializes the linear regression model
model = LinearRegression()

## Trains the logistic regression model using the training features which is X_train and the training labels y_train
model.fit(X_train, y_train)

## Make predictions as using the trained model to predict the median house value on the test set.
## This uses the trained model to predict the next day's temperature for the test set
y_pred = model.predict(X_test)

## Evaluate the model based on our training and testing using MSE and R2 score
## Calculates the mean_squared_error (MSE), which measures the average squared difference between the actual and the predicted temperature. A lower MSE indicates better performance,
## r2_score indicates how well the model fits the data. r2_score closer to one means better performance.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

## Visualize the data - Fig size  with width 10 and height 6.
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual Temperatures", marker='o')
plt.plot(y_pred, label="Predicted Temperatures", marker='x')
plt.title("Actual vs Predicted Temperatures")
plt.xlabel("Test Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.show()

## If we want to predict temperature for new data, we can pass a new data object
new_data = pd.DataFrame({
    'Temperature': [30],
    'Humidity': [60],
    'Wind Speed': [10],
    'Precipitation': [0],
})

## Predict tomorrow's temperature
predicted_temperature = model.predict(new_data)

## Print result
print(f"\n\nPredicted Temperature: {predicted_temperature[0]:.2f}C")
