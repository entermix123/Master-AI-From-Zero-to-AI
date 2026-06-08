# inport libraries

## Pandas is used for data manipulation and handling time series data.
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Get the model and implemented the linear regression algorithm to predict house prices.
from sklearn.linear_model import LinearRegression

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model
from sklearn.metrics import mean_squared_error, r2_score

## Plotting the actual predicted stock prices
import matplotlib.pyplot as plt

## Sample historical stock price data - we can use used stock prices from Yahoo Finance API or Alpha Vantage
data = {
    # generates a date range for ten consecutive days starting from January 1st 2024
    'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    # Prices for 10 days
    'Close': [150, 152, 153, 155, 154, 156, 157, 158, 159, 160]
}

## Create a data frame PD dot data frame and pass the data
## Convert the data set into a data frame
df = pd.DataFrame(data)

## Convert that date column to a numerical format or ordinal for regression analysis. his transformation is needed because the linear regression model works with numerical input.
## Converted into an ordinal format
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

## Display the first few rows of the data set
print("Stock Price Data:")
print(df.head())

## Create features and target variables
X = df[['Date']]
y = df['Close']

## Splits data into training and testing sets
## Split the data set using the train test split, which splits the data set into 80% for training and 20% for testing, and also will give a random state for it, which is one second
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

## Make predictions on the test set
y_pred = model.predict(X_test)

## Evaluate the model based on our training and testing using MSE and R2 score
## Calculates the mean_squared_error (MSE), which measures the average squared difference between the actual and the predicted temperature. A lower MSE indicates better performance,
## r2_score indicates how well the model fits the data. r2_score closer to one means better performance.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

## Print them and check it out how good they are
print(f"\nMean Squared Error: {mse}")
print(f"\nR2 Error: {r2}")

## Plot the actual versus predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, label="Actual Stock Prices", marker='o')
plt.plot(X_test, y_pred, label="Predicted Stock Prices", marker='x')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date (Ordinal)")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.show()

## Check the test the model with a new data set or a future date
future_date = pd.Timestamp('2024-01-11').toordinal()

## Create a data frame with the feature name date
future_date = pd.DataFrame({'Date': [future_date]})

## Predict the stock price for the future date
predicted_price = model.predict(future_date)

## Print prediction for future date
print(f"Predicted Stock Price: ${predicted_price[0]:.2f}")
