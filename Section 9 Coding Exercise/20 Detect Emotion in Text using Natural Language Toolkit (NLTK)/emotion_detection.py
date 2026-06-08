# import libraries

## Natural Language Toolkit
import nltk

## Stopwords which provides a list of common stopwords like 'and', 'the' which are filtered out to focus on the important words
from nltk.corpus import stopwords

## Tokenizes the text into individual words for further processing
from nltk.tokenize import word_tokenize

## Frequency distribution, useful for analyzing word frequencies through, though not essential for basic sentiment analysis, but still good
from nltk.probability import FreqDist

## Download required NLTK data (only needed once)
## Vader sentiment analysis tool specifically tuned for analyzing sentiment in social media text.
nltk.download("vader_lexicon")      # set as a comment after first download
## Stopwords are the common stop words used in the language
nltk.download("stopwords")          # set as a comment after first download
## Punct is the data used by NLTK for tokenizing sentences and words
nltk.download("punkt")              # set as a comment after first download

## Provides a free kind of pre-trained model for sentiment analysis, tuned for social media and informal text.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Get sample text for emotion detection I'm going to just add some text here
text = """
I am so happy today! The weather is butiful and everything is going well. I feel very positive and motivated!
"""

# Function to detect emotion in text
def detect_emotion(text):
    # Analyze sentiment
    ## This returns a dictionary of sentiment scores for the text including it will say positive, neutral, negative and compound, which is the overall sentiment score ranging from minus one to very negative to plus one which is very positive
    scores = sid.polarity_scores(text)

    # Display sentiment scores
    print("Sentiment Scores:", scores)

    # Determine emotion based on scores
    if scores["compound"] >= 0.5:
        emotion = "Joy"
    elif scores["compound"] <= -0.5:
        emotion = "Sadness"
    elif scores["neg"] > 0.5:
        emotion = "Anger"
    elif scores["neu"] > 0.7:
        emotion = "Neutral"
    else:
        emotion = "Mixed emotions"

    return emotion

# Detect and print the emotion
emotion = detect_emotion(text)
print("Detected Emotion:", emotion)