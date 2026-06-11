# Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

## converts text data, which is the movie genres, into numerical vectors using Tfidf (frequency, inverse document frequency) which helps wave words by importance in a document.
from sklearn.feature_extraction.text import TfidfVectorizer

## Using custom dataset for this example
data = {'movie_id': [1, 2, 3, 4, 5],
        'title': ['The Matrix', 'John Wick', 'The Godfather', 'Pulp Fiction', 'The Dark Knight'],
        'genre': ['Action, Sci-Fi', 'Action, Thriller', 'Crime, Drama', 'Crime, Drama', 'Action, Crime, Drama']
        }

## Convert the data set into a data frame
df = pd.DataFrame(data)

## Display the dataset
print(f"Movie Data: ")
print(df)

## Define a tf IDF Vectorizer advisor to transform the genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')

## Fit and transform the genre column into matrix of TF-IDF features
## Convert the genre column into a tf IDF matrix
## Each row in this particular represents a movie and each column represents a genre related word.
## The values are the TF-IDF scores, which way the weight of the word depending on how much is repeated, indicating how important each word is in each movie genre description.
tfidf_matrix = tfidf.fit_transform(df['genre'])

## Compute the cosine similarity matrix
## This returns a cosine similarity matrix where each element, which is each tuple I and j represents the similarity between the movie I and movie j, so it kind of cross-references them. That's why I gave the same matrix twice.
consine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on cosine similarity. Six steps that will follow here
## First we'll get the index of the movie that matches the title.
def get_recommendations(title, cosine_sim=consine_sim):
    # Get the index of the movie taht matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pair wise similarity Of course of all movies with that movie.
    # It returns the similarity scores between the input movies and all other movies.
    # The result is a list of tuples where each tuple contains a movie index and its similarity score.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores.
    # Sort the movies by similarity in descending order of most similar first.
    # And once we have that we can get the indices of the two most similar movies in this particular case.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the two most similar movies
    sim_scores = sim_scores[1:3]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the most similar movies
    return df['title'].iloc[movie_indices] # type: ignore

# Test the recommendation system with an example
movie_title = 'The Matrix'
recommended_movies = get_recommendations(movie_title)

print(f"Movie recommended for '{movie_title}':")
for movie in recommended_movies:
    print(movie)