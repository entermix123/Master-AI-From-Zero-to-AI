# import libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Implements the Random Forest Regressor model for predicting prices
from sklearn.ensemble import RandomForestRegressor

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model.
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (replace with a real dataset if available)
data = {
    'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
    'model': ['Corolla', 'Civic', 'F-150', '3 Series', 'A4', 'Camry', 'Accord', 'Mustang', '5 Series', 'A6'],
    'year': [2015, 2017, 2018, 2016, 2015, 2018, 2016, 2015, 2017, 2018],
    'mileage': [50000, 30000, 40000, 60000, 45000,35000, 55000, 65000, 30000, 20000],
    'price': [15000, 17000, 25000, 27000, 30000, 16000, 18000, 26000, 28000, 31000]
}

# Convert dataset into DataFrame
df = pd.DataFrame(data)
print("Dataset:\n, df")

# Convert categorical columns to numeric using one-hot encoding
## Encode the categorical features using one hot encoding because we have some text over there and we want to convert it into numbers.
df_encoded = pd.get_dummies(df, columns=['make', 'model'])
## df_encoded will give us the encoded numbers. So we print it out and see the values.
## df_encoded is the resulting DataFrame, where each unique value is make and model, has its own column with the binary indication one if the value is present and zero otherwise.
print("\nEncoded Dataset:\n", df_encoded)

# Define features (X) and target (y)
## X is a DataFrame containing all columns except the target variable price representing the features used for training. And Y is the price column which we want to predict.
## We want to use price as target and not features. So we are dropping that here.
X = df_encoded.drop('price', axis=1)  # Features
## In target we including the price which is target variable.
y = df_encoded['price']               # Target variable

# Split the dataset into training and testing sets
## Split the data set using the train test split, which splits the data set into 80% for training and 20% for testing, and also will give a random state for it, which is one second
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
## n_estimators=100 specifies 100 trees in the forest, which is a typical starting point
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
## model.fit trains the random forest model on the training set, X train and Y train. The model learns the relationship between the features and the target variable.
model.fit(X_train, y_train)

# Make predictions on the test set
## model.predict uses the trained model to predict car prices for the test set, which is X_test and y_pred will finally store the predicted values.
y_pred = model.predict(X_test)

# Evaluate the model
## Calculates the mean squared error MSE between the actual and the predicted car prices. MSE represents the average squared difference between predicted and actual values.
mse = mean_squared_error(y_test, y_pred)
## r2_score calculates the r squared score, which indicates how well the model explains the variance in the target variable, r2 ranges from 0 to 1, and one indicates a perfect fit.
r2 = r2_score(y_test, y_pred)

# Print results
## We can check if MSE a lower value indicates better prediction accuracy for MSE, whereas r2 score of variable value in closer to one indicates the model explains most of the variance in the data.
print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")