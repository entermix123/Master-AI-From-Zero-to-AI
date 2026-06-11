# Import libraries

## NLTK is natural language toolkit, as I mentioned, which provides tools for text processing and classification
import nltk

## corpus contains labeled movie reviews positive and negative And we'll use this data set for training and testing
from nltk.corpus import movie_reviews

## Naive Bayes classifier - probabilistic classifier based on Bayes theorem commonly used for text classification tasks like sentiment analysis
from nltk.classify import NaiveBayesClassifier

## Accuracy is used to evaluate the accuracy of this classifier
from nltk.classify.util import accuracy as nltk_accuracy

## Contains common stop words like 'the', 'and' that are usually removed during text processing.
from nltk.corpus import stopwords

## shuffle the data set and ensure randomness
import random

## Download NLTK data files
nltk.download('moview_reviews')

## Punct is a tokenizer for splitting sentences into words
nltk.download('punkt')

## We'll also download the stopwords.
nltk.download('stopwords')

## Pre-process the dataset and extract the features
## This function converts a list of words into a dictionary, where the keys are the words and the values are true
## This format is required by NLTK classifiers to process the features.
def extract_features(words):
    return {word: True for word in words}

## Load and preprocess the dataset (load movie_reviews dataset from NLTK)
## movie_reviews.words(fileid) - loads the loads the words from each movie review in the dataset
## moview_reviews.categories() - returns the two categories, which is positive, which pos for positive and neg for negative
## movie_reviews.fileids(category) - represents a specific movie review file
## So this creates a list of tuples where each tuple contains a list of movie reviews, file id the words in the review and the category is is the corresponding label either positive for pos for positive or neg for negative.
documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

## Shuffle the dataset to ensure random distribution
random.shuffle(documents)

## Prepare the dataset for taining and testing
## Converts each review into a set of features using the extract feature function
featuresets = [(extract_features(d), c) for (d,c) in documents]

## create train_set and test_set, which we take from feature sets
## out of the total 2000, we have divided into 80% for train and 20% for test
train_set, test_set = featuresets[:1600], featuresets[1600:]

## Train the the Naive Bayes classifier using the training set, the classifier learns to distinguish between positive and negative sentiments based on the words in these reviews
classifier = NaiveBayesClassifier.train(train_set)

## Evaluate the classifier on the test set
## Classifier test set on line number 35 evaluates the classifier's accuracy on the test set.
## This function compares the classifiers predicted labels with the actual labels in the test set, and calculates the percentage of correct predictions.
accuracy = nltk_accuracy(classifier, test_set)

## Print the accuracy as a percentage
print(F"Accuracy: {accuracy * 100:.2f}%")

## Show the most informative features - display the ten top ten words that provide the most information for distinguishing between positive and negative reviews. The words that the classifier finds most useful in making its prediction.
classifier.show_most_informative_features(10)

## Test on the new input sentences
def analyze_sentient(text):
    # Tokanize the input text into words
    words = nltk.word_tokenize(text)
    # Remove common stop words like the and, etc. from the English dictionary
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    # predict the sentiment
    features = extract_features(words)
    # classify either positive or negative for based on the features
    return classifier.classify(features)

# Test the classifier with custom text input
test_sentences = [
    "This movie is absolutely fantastic! The acting, the story, everything was amazing!",
    "I hated this movie. It was a waste of time and money.",
    "The plot was a bit dull, but the performance were great.",
    "I have mixed feelings about this film. It was okay, not great but not terrible either.",
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {analyze_sentient(sentence)}")
    print()