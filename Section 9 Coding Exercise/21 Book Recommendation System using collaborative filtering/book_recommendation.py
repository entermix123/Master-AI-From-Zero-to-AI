# Import necessary libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Calculates the cosine similarity between vectors, which we use to compute similarity between books based on user ratings
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of user ratings (replace with a real dataset if available)
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'book_title': [
        'Book A', 'Book B', 'Book C', 'Book A', 'Book D',
        'Book B', 'Book C', 'Book E', 'Book A', 'Book C'
    ],
    'rating': [5, 3, 4, 4, 5, 5, 3, 4, 3, 2]
}

# Convert dataset to DataFrame
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Create a user-book matrix
## df.pivot_table transforms the data set into a user book matrix, where each column is a book and the values are the ratings.
user_book_matrix = df.pivot_table(index='user_id', columns='book_title', values='rating').fillna(0) 
print("\nUser-Book Matrix:\n", user_book_matrix)

# Calculate cosine similarity between users
## Calculate the cosine similarity between books based on the user ratings in the in the user book matrix.
user_similarity = cosine_similarity(user_book_matrix)
## We create a data frame using the user similarity. The index is equal to user book matrix, dot index and columns equal to user book matrix dot index.
user_similarity_df = pd.DataFrame(user_similarity, index=user_book_matrix.index, columns=user_book_matrix.index)
## Print the user similarity matrix
print("\nUser Similarity Matrix:\n", user_similarity_df)


# Function to recommend books based on user similarity
def recommend_books(user_id, similarity_matrix, user_book_matrix, top_n=3):

    if user_id not in similarity_matrix.index:
        print("User not found in the dataset.")
        return []

    # Get similarity scores for the user
    similar_users = similarity_matrix[user_id].sort_values(ascending=False).drop(user_id)

    # Aggregate ratings from similar users, weighted by similarity
    ## create an empty dictionary of recommended books
    recommended_books = {}
    for sim_user, similarity in similar_users.items():
        ## create a variable called rated_books and check the assign user book location based on the similar user
        rated_books = user_book_matrix.loc[sim_user]
        ## Go over each and every rated books where rated books is greater than zero. We are going to get the book and rating and check if the book is not in user book metrics.
        for book, rating in rated_books[rated_books > 0].items():
            if book not in user_book_matrix.loc[user_id] or user_book_matrix.loc[user_id, book] == 0:
                recommended_books[book] = recommended_books.get(book, 0) + rating * similarity

    # Sort books by aggregated score and return top recommendations
    recommended_books = sorted(recommended_books.items(), key=lambda x: x[1], reverse=True)
    ## Return the book for book score in recommended books back to the person who actually the function that called it.
    return [book for book, score in recommended_books[:top_n]]

# Get recommendation for a specific user
user_id = 1
# user_id = 4
recommended_book = recommend_books(user_id, user_similarity_df, user_book_matrix, top_n=3)
print(f"\nBooks recommended for User {user_id}:", recommended_book)