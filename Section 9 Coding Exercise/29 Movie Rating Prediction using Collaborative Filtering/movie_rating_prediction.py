# import libraries

## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## matrix operation and mathematical function
import numpy as np

## This is singular value decomposition algorithm from the surprise library used for collaborative filtering
from surprise import SVD

## Data set is used to build a data set from raw data and reader helps defines the rating scale and data set structure.
from surprise import Dataset, Reader

## train_test_split splits data set into training and testing sets
from surprise.model_selection import cross_validate, train_test_split

## Contains metrics for example RMSE for evaluating models
from surprise import accuracy

## Visualization library for creating statistical plots
import seaborn as sns

## Core plotting library for visualization
import matplotlib.pyplot as plt

# Step 1: Load Dataset
# Load MovieLens dataset
df = pd.read_csv('data/ratings.csv')
## We are going to drop timestamp as is not required. This removes the timestamp column as it's irrelevant for collaborative filtering.
df.drop('timestamp', axis=1, inplace=True)

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Step 2: Preprocessing the Dataset
# Define a Reader object for Surprise
## Defines the range of ratings in the data set
reader = Reader(rating_scale=(1, 5))
## Load the data set into a format compatible with surprise. Input equal to user ID, movie ID, rating and surprise expects a data set with three columns user ID, item ID and rating.
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Step 3: Split Data into Training and Testing Sets
## train_test_split splits the data into training and testing sets
## test_size=0.2 Reserves 20% of the data for testing
## random_state=42 ensures reproducibility - Every time you use this number, it will give the exact same training set and test set
## trainset that we get out of here will be the training data used for train to train
## testset - testing data used for evaluation
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Step 4: Build Collaborative Filtering Model using SVD
## The SVD model predicts ratings for user item pairs based on the latent factors
## Initializes the singular value decomposition or SVD model
model = SVD()
## Train the model on the training data set
model.fit(trainset)

# Step 5: Evaluate the Model
## Tests the model on test dataset and generates predictions
predictions = model.test(testset)
## RMSE measures the difference between predicted and actual ratings, and lower RMSE indicates better performance
## Calculates the root mean square error which is RMSE
## predictions is a list of predicted ratings for the user input pairs that we get from the command above
rmse = accuracy.rmse(predictions)
## print RMSE so we can see how it looks like
print(f"RMSE: {rmse:.4f}")

# Step 6: Make a Prediction for a Specific User and Movie
user_id = 196
movie_id = 242

## predict the rating for specific user
predicted_rating = model.predict(user_id, movie_id).est
## print predicted rating for the user
print(f"Predicted Rating for User {user_id} on Movie {movie_id}: {predicted_rating:.2f}")

# Step 7: Visualize Distribution of Ratings
## Set the plot size
plt.figure(figsize=(10, 6))
## Creates a histogram of movie ratings
## bins=5 divides the rating into five different groups
## kde=True displays a smooth kernel density estimate
sns.histplot(df['rating'], bins=5, kde=True)
## Set plot title
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
## print the plot
plt.show()
