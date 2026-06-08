# import Libraries

## Natural Language Toolkit
import nltk

## Stopwords which provides a list of common stopwords like 'and', 'the' which are filtered out to focus on the important words
from nltk.corpus import stopwords

## Word tokenize and send tokenize which will tokenize our words and sentences. NLTK tokenize provides functions for tokenizing the text into sentences and words
from nltk.tokenize import word_tokenize, sent_tokenize

# Download the Stopword (only needed once)
## Stopwords are the common stop words used in the language
nltk.download("stopwords")          # set as a comment after first download
## Punct is the data used by NLTK for tokenizing sentences and words
nltk.download("punkt")              # set as a comment after first download

## Define the text for summarization
text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal.
A subset of artificial intelligence is machine learning, which refers to the concept that computer programs can automatically learn from and adapt to new data without being explicitly programmed.
Deep learning techniques enable this automatic learning through the absorption of huge amounts of unstructured data such as text, images, or video.
"""

# Function to generate a frequency-based summary
def summarize_text(text, num_sentences=2):
    # Tokenize text into sentences and words
    # splits the text into individual sentences
    sentences = sent_tokenize(text)             
    # tokenizes the text into words and converts them to lowercase to ensure consistent counting
    words = word_tokenize(text.lower())         

    # filter out the stopwords and non-alphabetical words
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}

    # dictionary that um that stores the frequency of each non stop word Non-numeric word in the text. So only words that are alphabetic.
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1


    # score each sentence based on the word frequency
    ## dictionary that stores a score for each sentence based on the frequencies of the words it contains
    ## So sentences with frequently occurring words are given higher scores
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    # Sort the sentences by score and select the top 'num_sentences'
    ## create a summary sentences called the sorted function in sentences score and key equal to sentence score dot get and reverse text
    ## sorts the sentences by score in descending order and selects the top number of sentences
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    ## create the summary, which I will join the summary sentences together in a single single word
    summary = " ".join(summary_sentences)
    ## return back the summary that we have created in this particular function
    return summary

# Generate and print by saying summary 
summary = summarize_text(text, num_sentences=2)
print("Original Text:\n", text)
print("\nSummary:\n", summary)