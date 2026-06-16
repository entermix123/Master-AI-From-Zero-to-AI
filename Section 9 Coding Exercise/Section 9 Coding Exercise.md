# Section 9 Coding Exercise

## Content
53. [1: Basic Calculator using Python](#53-1-basic-calculator-using-python)
54. [2: Image Classifier using Keras and TensorFlow](#54-2-image-classifier-using-keras-and-tensorflow)
55. [3: Simple Chatbot using predefined responses](#55-3-simple-chatbot-using-predefined-responses)
56. [4: Spam Email Detector using Scikit-learn](#56-4-spam-email-detector-using-scikit-learn)
57. [5 Human Activity Recognition](#57-5-human-activity-recognition)
58. [6: Sentiment Analysis on text data using NLTK](#58-6-sentiment-analysis-on-text-data-using-nltk)
59. [7: Movie Recommendation System using cosine similarity](#59-7-movie-recommendation-system-using-cosine)
60. [8: Predict House Prices with Linear Regression](#60-8-predict-house-prices-with-linear-regression)
61. [9: Weather Forecasting using historical data](#61-9-weather-forecasting-using-historical-data)
62. [10: Basic Neural Network from scratch](#62-10-basic-neural-network-from-scratch)
63. [11: Stock Price Prediction using historical data w/ simple Linear Regression](#63-11-stock-price-prediction-using-historical-data-w-simple-linear-regression)
64. [12: Predict Diabetes using logistic regression](#64-12-predict-diabetes-using-logistic-regression)
65. [13: Dog vs. Cat Classifier with CNN](#65-13-dog-vs-cat-classifier-with-cnn)
66. [14: Tic-Tac-Toe AI using Minimax Algorithm](#66-14-tic-tac-toe-ai-using-minimax-algorithm)
67. [15: Credit Card Fraud Detection using Scikit-learn](#67-15-credit-card-fraud-detection-using-scikit-learn)
68. [16: Iris Flower Classification using decision trees](#68-16-iris-flower-classification-using-decision-trees)
69. [17: Simple Personal Assistant using Python speech libraries](#69-17-simple-personal-assistant-using-python-speech-libraries)
70. [18: Text Summarizer using Gensim](#70-18-text-summarizer-using-gensim)
71. [19: Fake Product Review Detection using NLP techniques](#71-19-fake-product-review-detection-using-nlp-techniques)
72. [20: Detect Emotion in Text using Natural Language Toolkit (NLTK)](#72-20-detect-emotion-in-text-using-natural-language-toolkit-nltk)
73. [ 21: Book Recommendation System using collaborative filtering](#73-21-book-recommendation-system-using-collaborative-filtering)
74. [74. 22: Predict Car Prices using Random Forest](#74-22-predict-car-prices-using-random-forest)
75. [23: Identify Fake News using Naive Bayes](#75-23-identify-fake-news-using-naive-bayes)
76. [24: Create a Resume Scanner using keyword extraction](#76-24-create-a-resume-scanner-using-keyword)
77. [25: Customer Churn Prediction using classification algorithms](#77-25-customer-churn-prediction-using-classification-algorithms)
78. [26: Named Entity Recognition (NER) using spaCy](#78-26-named-entity-recognition-ner-using-spacy)
79. [27: Predict Employee Attrition using XGBoost](#79-27-predict-employee-attrition-using-xgboost)
80. [28: Disease Prediction (e.g., Heart Disease) using ML algorithms](#80-28-disease-prediction-eg-heart-disease-using-ml-algorithms)
81. [29: Movie Rating Prediction using Collaborative](#81-29-movie-rating-prediction-using-collaborative-filtering)
82. [ 30: Automatic Essay Grading using BERT](#82-30-automatic-essay-grading-using-bert)

## 53. 1: Basic Calculator using Python

[⬆ Back to content](#content)

This project will involve implementing a simple command line calculator that can perform basic arithmetic operations such as addition, subtraction, multiplication, and division.

calc.py 

```python
# add function
def add(x, y):
    return x + y

# subtract function
def subtract(x, y):
    return x - y

# multiply function
def multiply(x, y):
    return x * y

# divide function
def divide(x, y):
    if y == 0:
        return "Error! Division by zero not allowed."
    else:
        return x / y

def calculator():
    print("Select operation:")
    print("1: Add")
    print("2: Subtract")
    print("1: Multiply")
    print("1: Divide")

    while True:
        # Take input from the user
        choice = input("Enter choices (1.2.3.4): ")

        # Check if the input is correct
        if choice in ['1', '2' ,'3' , '4']:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))

            if choice == '1':
                print(f"{num1} + {num2} = {add(num1 , num2)}")
            
            if choice == '2':
                print(f"{num1} - {num2} = {subtract(num1 , num2)}")

            if choice == '3':

                print(f"{num1} * {num2} = {multiply(num1 , num2)}")

            if choice == '4':
                print(f"{num1} / {num2} = {divide(num1 , num2)}")

        # Option to exit the loop
        next_calculation = input ("Do you want to perform another calculation? (yes/no)")
        if next_calculation.lower() != 'yes':
            break

        
    print("Exiting Calculator. Goodbye!")

# Call calculator function
calculator()
```
Run the application and test it
    terminal --> python calc.py 

[⬆ Back to content](#content)


## 54. 2: Image Classifier using Keras and TensorFlow

[⬆ Back to content](#content)

This project involves building a simple image classifier to recognize handwritten digits using the mNIST dataset, which is a data set of 28 by 28 grayscale images of handwritten digits from 0 to 9.

Steps:
1: Install Required Libraries
2: Load the MNIST Dataset
3: Preprocess the Data
4: Build the model
5: Train the model
6: Evaluate and Make Predictions

1: Install Required Libraries
pip install tensorflow keras

Create file classifier.py

```python
## matrix operation and mathematical function
import numpy as np

## TensorFlow is a deep learning library
## Here we use it to build and train the CNN
import tensorflow as tf 

## TensorFlow Keras is Keras which is included with TensorFlow simplifies the neural network creation. 
## We are importing data sets which are the module for loading popular data sets, including mNIST.
## We have layers which is used to add layers to our CNN
## We have models which use to define and compile a neural network model
from tensorflow.keras import datasets, layers, models

## two categorical converts labels to one hot encoded format, which is necessary for classification tasks.
from tensorflow.keras.utils import to_categorical

## To see this data in real time and see the image of which it predicted
import matplotlib.pyplot as plt

# 2: Load the MNIST Dataset
## Keras comes with built in datasets, including mNIST, making it very easy to load the data
## This function it will give us two different tuples which will have train images and train labels and test images and test labels
## train images are the training data and corresponding labels which are digits 0 to 9
## second tuple that we get is as I mentioned is test images and test labels. Now these are the testing data and labels for evaluating model accuracy.
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


# 3: Preprocess the Data
## Data needs to be normalized to improve the model's performance
## Preprocessing: normalize the pixel values to be between 0 and 1
## So what here normalization means we divide each pixel value by 255 to scale the values between 0 and 1, making the training faster and more stable
train_images = train_images / 255.0
test_images = test_images / 255.0

## Reshape images to (28, 28, 1) as they are grayscale
## CNN expects each image in 3D shape which is height, width and channels. Since mNIST images are grayscale, each has only one channel, and that's why 28 28 1 which represents height, width and one color channel
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

## Convert the labels to one hot encoded format. Now I we do this using the two categorical that we have imported, which converts labels Example 3 to 1 hot vectors. Example 000100000 for classifying 0 to 9. Now the purpose of this is this format is required for multi-class classification with categorical cross entropy loss.
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 4: Build the CNN model
## We'll use a simple feed forward neural network which is fully connected layers for classification
## Model dot sequential initializes a sequential model which allows layers to be stacked linearly
model = models.Sequential()

## Build the first convolution layer
## calling this conv 2D 30 233 activation which adds a 2D convolution convolutional layer where 32 is the number of filters to apply three by three is the size of each filters, which is three by three pixels, and activation equal to ReLU is the activation function applied to each pixel, setting negative values to zero, and then finally input shape 2828 one, which specifies the input dimension for the first layer.
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

## Creates the first convolutional layer
## reduces the dimensions of the output by taking the maximum value in each two by two block, which decreases computational load and helps detect important features
model.add(layers.MaxPooling2D((2,2)))

## Creates the second convolutional layer
## The second convolutional layer first we call that conv 2d function which adds another convolutional layer with 64 filters
## The max pooling 2D reduces the output dimension by taking the maximum value in each two by two block
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

## Creates the third convolutional layer
## No pooling is applied after this layer, allowing it to like preserve the spatial dimensions for flattening, which means we are we have to flatten
model.add(layers.Conv2D(64, (3,3), activation='relu'))

## Flatten the 3D aoutput to 1D and add Dense layer
## flatten converts the 3D output of the previous layer to a 1D vector, enabling it to connect to fully connected dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

## Add a dense layer - output layer with 10 neurons (for 10 digit classes)
## This will create the final layer with ten units for each digit which is 0 to 9. That's why I've used the number ten. And then I'm using softmax which converts the output to probabilities, helping the model classify each digit.
model.add(layers.Dense(10, activation='softmax'))

## Compile the model
## specifying the Adam optimizer for adaptive learning. Then for loss I'm saying categorical and cross entropy which has loss function for multi-class classification with one hot encoded labels and metrics equal to accuracy which tracks the model accuracy during training and testing.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5: Train the model
## We'll train the model on the training set and validate it on the test set
## Trains the model on training data. We have passed the training images and training labels, which are the training inputs and outputs that we created epochs equal to five. Now this is the number of training epochs, meaning the model will see the entire training data set five times next batch size equal to 64 number of samples processed before updating any model parameters
## Validation data which it can validate against. We have passed the test images and test labels, which validate which is our validation data used to evaluate model performance during the training process
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# 6: Evaluate and Make Predictions
## Evaluate the model on test data
## This function evaluates model performance on test data returning loss and accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

## show test_loss and test_accuracy
print(f"Test accuracy: {test_acc * 100:.2f}%")

## Make Predictions on test images
## the predict test images function here uses the trained model to predict class probabilities for each test image, and the np.argmax finds the index with the highest probability in the first prediction, which corresponds to the predicted digit
predictions = model.predict(test_images)
print(f"repdiction for first test image: {np.argmax(predictions[0])}")

plt.imshow(test_images[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted Label: {predictions[0].argmax()}")
plt.show()
```

Run the file
    terminal --> python classifier.py

<img src="2 Image Classifier using Keras and TensorFlow/pics/result.png" width="400" />
<br>
<br>


We can check the fourth image by modifing the last few commands in the prediction section
```python
...
print(f"repdiction for fourth test image: {np.argmax(predictions[4])}")

plt.imshow(test_images[4].reshape(28,28), cmap='gray')
plt.title(f"Predicted Label: {predictions[4].argmax()}")
plt.show()
```

Run the file again
    terminal --> python classifier.py

<img src="2 Image Classifier using Keras and TensorFlow/pics/result2.png" width="400" />
<br>
<br>


[⬆ Back to content](#content)


## 55. 3: Simple Chatbot using predefined responses

[⬆ Back to content](#content)

Simple chatbot in Python that responds with predefined answers based on keywords in user input

chatbot.py

```python
# Import regex
import re

# Mapped reposnses to keywords
responses = {
    "Hello": "Hi there! How can I help you today?",
    "hi": "Hello! How can I help you?",
    "how are you": "I am just a bot, but I am doing great! How about you?",
    "what is your name": "I'm a chatbot created to assist you",
    "help": "Sure, I am here to help. What do you need assistance with?",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're wellcome! I'm happy to help.",
    "default": "I'm not sure I uderstand. Could you please rephrase?"
}

# Funtion to find the appropriate response based on the user input
def chatbot_response(user_input):
    # Convert input to lowercase
    user_input = user_input.lower()

    for keyword in responses:
        if re.search(keyword, user_input):
            return responses[keyword]

    return responses["default"]

# Main function to run the chatbot
def chatbot():
    print(f"Chatbot: Hello! I'm here to assist you. (type 'bye' to exit)")

    while True:
        # Get user input
        user_input = input("You: ")

        # If user types 'bye', exit the loop
        if user_input.lower() == 'bye':
            print(f"Chatbot: Goodbye! Have a great day!")
            break

        # Get chatbot's response based on user inpput
        response = chatbot_response(user_input)

        # Print Chatbot's response
        print(f"Chatbot:{response}")

# Run Chatbot
chatbot()
```

Run the chatbot and test it
    terminal --> python chatbot.py

[⬆ Back to content](#content)


## 56. 4: Spam Email Detector using Scikit-learn

[⬆ Back to content](#content)

We will be building a machine learning model that classifies email as spam or not spam based on their text context.

By using techniques like text pre-processing logistic regression, we will learn to transform raw email data into features that the model can interpret.

The project demonstrates fundamental steps in text classification and equips you with the knowledge to build models that processes and interprets text data.

Download dataset from kaggle.com - https://www.kaggle.com/datasets/colormap/spambase - spam.csv

spam_detector.py

```python
# import libraries
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Binary classification problems in which in our case is spam versus not spam
from sklearn.linear_model import LogisticRegression

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## Precision score, which measures the proportion of positive predictions that are actually correct
## Recall score, which measures the proportion of actual positives that are correctly identified
## F1 score which is the harmonic mean of precision and recall
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

## data visualization - It is built on top of matplotlib and makes it easy to create beautiful and informative statistical graphics
import seaborn as sns

## allows us to plot graphs and charts such as the confusion matrix that we will do
import matplotlib.pyplot as plt

# Loading Dataset
## Split the dataset to training and testing datasets
data = pd.read_csv('spam.csv')

## drop the column named spam from the data set to create the feature matrix X
## X will contain all the columns except the target column spam
## This will be the data used to predict whether an email is spam or not
X = data.drop('spam', axis=1)

## y is the target variable, which is the spam column, which contains the binary labels zero for not spam and one for spam
y = data['spam']

# Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## Setting a random state ensures reproducibility
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the logistic regression model to classify emails as spam or not spam
## initialize logistic regression model
model = LogisticRegression()

## trains the logistic regression model using the training features which is X_train and the training labels y_train
model.fit(X_train, y_train)

## making predictions - uses the model to predict the labels spam or not spam nn the test set X_test, the predicted labels are stored in the variable y_pred
y_pred = model.predict(X_test)

## Optional - Print data
print(X_test)

## Evaluate the model using the accuracy, confusion matrix, precision, recall and F1 score
## calculating the accuracy of the model, which is the percentage of correct predictions both spam and not spam
accuracy = accuracy_score(y_test, y_pred)

## calculates precision, which is the proportion of emails predicted as spam that are actually spam, which is that is like true positives divided by true positives plus false positives
precision = precision_score(y_test, y_pred)

## calculates the recall which is the proportion of actual spam emails that are correctly predicted - true positives divided by true positives plus false negatives.
recall = recall_score(y_test, y_pred)

## harmonic mean of precision and recall. It provides a single metric that balances both precision and recall
f1 = f1_score(y_test, y_pred)

## Print variables
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")


# Visualize the confusion matrix using Seaborn heatmap
## generates the confusion matrix a table that shows it will show me four different sections
## One is true positive which is emails that were correctly predicted as spam. 
## Two is true negatives which are emails that were correctly predicted as not spam. 
## Tree is false positives - emails that were incorrectly predicted spam but were not spam. 
## Four - false negatives which are emails that were incorrectly predicted as not spam but were actually spam.
cm = confusion_matrix(y_test, y_pred)

## visualize the confusion matrix as heatmap 
## annot=True - annotates the heatmap with the actual values from the confusion matrix
## fmt='d' - ensures that the values are shown as integers and not scientific notation
sns.heatmap(cm, annot=True, fmt='d')

# Set title and print the heatmap
plt.title('Confusion Matrix')
plt.show()
```

Run the file      
    terminal --> python spam_detector.py      

<img src="4 Spam Email Detector using Scikit-learn/pics/spam-detector-results-2.png" width="1000" />
<br>
<br>
[921 rows x 57 columns]     
Accuracy: 0.9250814332247557        
Precision: 0.9212598425196851       
Recall: 0.9     
F1-Score: 0.9105058365758755        
  
<img src="4 Spam Email Detector using Scikit-learn/pics/spam-detector-results.png" width="600" />
<br>
<br>

[⬆ Back to content](#content)


## 57. 5: Human Activity Recognition

[⬆ Back to content](#content)

Human activity recognition using smartphones. Data set with random forest. This project will classify human activities like walking, sitting, standing, etc. based on the sensor data from the smartphones.

We will use the human Activity Recognition which is Har data set from UCI, which contains data from accelerometers and gyroscopes in smartphones. The project will use Random Forest as the classifier.

Download dataset from kaggle.com - https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones, Unzip and rename train.csv to har_data.csv


human_activity_recognition.py

```python
# import libraries
## pandas used for loading and manipulating the data set
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## random forest classifier - used for human activity recognization
from sklearn.ensemble import RandomForestClassifier

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## Precision score, which measures the proportion of positive predictions that are actually correct
## Recall score, which measures the proportion of actual positives that are correctly identified
## F1 score which is the harmonic mean of precision and recall
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

## data visualization - create a heat map from the confusion matrix visualization
import seaborn as sns

## plotting the confusion matrix
import matplotlib.pyplot as plt

# Load dataset - https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
## loads the Human Activity Recognition which has HR data set
## The data set contains sensor data from accelerometer and gyroscope, and corresponding activity labels will have like walking, sitting, standing, all that stuff inside this data set
df = pd.read_csv('har_data.csv')

## Pre-process the dataset
## Drop the column that I don't need - removes the activity column from the data set leaving only the features which are the sensor readings
X = df.drop('Activity', axis=1)

## target variable contains the activity labels - example walking, sitting and standing.
y = df['Activity']

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target 
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## Setting a random state ensures reproducibility
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Train the RandomForestClassifier
## The random forest is an ensemble method that uses multiple decision trees for classification. That's why we have provided the 100 value to it.
model = RandomForestClassifier(n_estimators=100, random_state=42)

## Train the model
model.fit(X_train, y_train)

## Make predictions on the test set
y_pred = model.predict(X_test)

## Evaluate the model using accuracy, precision, recall, f1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

## Print variables
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_Score: {f1 * 100:.2f}%")

## Visualize the confusion matrix using Seaborn's heatmap
## generates the confusion matrix a table that shows it will show me four different sections
## One is true positive 
## Two is true negatives 
## Tree is false positives
## Four - false negatives
cm = confusion_matrix(y_test, y_pred)

## visualize the confusion matrix as heatmap 
## annot=True - annotates the heatmap with the actual values from the confusion matrix
## fmt='d' - ensures that the values are shown as integers and not scientific notation
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set title and print the heatmap
plt.title('Confusion Matrix')
plt.show()
```

Run the file      
    terminal --> python human_activity_recognition.py   

Accuracy: 98.03%        
Precision: 98.03%       
Recall: 98.03%      
F1_Score: 98.03%        

<img src="5 Human Activity Recognition/pics/human_activity_results.png" width="600" />
<br>
<br>

[⬆ Back to content](#content)


## 58. 6: Sentiment Analysis on text data using NLTK

[⬆ Back to content](#content)

We will implement a sentiment analysis on text data using NLTK, which stands for Natural Language Toolkit.

In this project, we will classify the sentiment of the text data, whether it's positive or negative, based on the text content.

We will use the NLTK library to preprocess the data and a Naive Bayes classifier to perform the sentiment analysis.

sentiment_analysis_with_natural_langauge.py

```python
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
```

Run the file      
    terminal --> python sentiment_analysis_with_natural_langauge.py  

Result:
Accuracy: 70.75%        
Most Informative Features       
               insulting = True              neg : pos    =     15.3 : 1.0      
             outstanding = True              pos : neg    =     13.8 : 1.0      
              vulnerable = True              pos : neg    =     12.7 : 1.0      
               ludicrous = True              neg : pos    =     12.1 : 1.0      
                chilling = True              pos : neg    =     11.4 : 1.0      
                  turkey = True              neg : pos    =     11.3 : 1.0      
            effortlessly = True              pos : neg    =     10.8 : 1.0      
                 insipid = True              neg : pos    =     10.6 : 1.0      
                  seagal = True              neg : pos    =     10.6 : 1.0      
               stupidity = True              neg : pos    =     10.4 : 1.0      
Sentence: This movie is absolutely fantastic! The acting, the story, everything was amazing!        
Predicted sentiment: pos        

Sentence: I hated this movie. It was a waste of time and money.     
Predicted sentiment: neg        

Sentence: The plot was a bit dull, but the performance were great.      
Predicted sentiment: neg        

Sentence: I have mixed feelings about this film. It was okay, not great but not terrible either.        
Predicted sentiment: neg        

[⬆ Back to content](#content)


## 59. 7: Movie Recommendation System using cosine 

[⬆ Back to content](#content)

We will build a movie recommendation system using cosine similarity. 

The idea behind this project is to recommend movies based on how similar they are to a given movie. 

Using cosine similarity to measure the similarity between the two movie feature vectors.

movie_recommender.py

```python
# Import Libraries
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Calculates the cosine similarity between vectors
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
    return df['title'].iloc[movie_indices]

# Test the recommendation system with an example
movie_title = 'The Matrix'
recommended_movies = get_recommendations(movie_title)

print(f"Movie recommended for '{movie_title}':")
for movie in recommended_movies:
    print(movie)
```

Run the file        
    terminal --> python movie_recommender.py

Result:     
Movie Data:         
   movie_id            title                 genre      
0         1       The Matrix        Action, Sci-Fi      
1         2        John Wick      Action, Thriller      
2         3    The Godfather          Crime, Drama      
3         4     Pulp Fiction          Crime, Drama      
4         5  The Dark Knight  Action, Crime, Drama      
Movie recommended for 'The Matrix':     
The Dark Knight     
John Wick       


[⬆ Back to content](#content)


## 60. 8: Predict House Prices with Linear Regression

[⬆ Back to content](#content)

We will use the California Housing data set to predict house prices with linear regression.

This data set contains information about different districts in California and includes features such as population, median income, house age, and more.

predict_house_prices.py

```python
# Import libraries
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## matrix operation and mathematical function
import numpy as np

## Load data
from sklearn.datasets import fetch_california_housing

## Splits the data set into training and testing sets
from sklearn.model_selection import train_test_split

## Get the model and implemented the linear regression algorithm to predict house prices.
from sklearn.linear_model import LinearRegression

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model
## mean_squared_error measures the mean squared error of prediction
## r2_score measures the proportion of variance explained by the model
from sklearn.metrics import mean_squared_error, r2_score

## Loads the California housing data sets and returns it as a pandas data frame.
## This data set contains features like median income, house age, average number of rooms, and others, along with the target variable, which is the median house value, which is actually in hundred of thousands of dollars. If its 1 it means it's 100,000, if it's 1.5, it means 1.5 million.
housing = fetch_california_housing(as_frame=True)

## Create a Dataframe from the dataset
## Convert the data set into a data frame for easy manipulation
df = housing.frame

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
```

Run the file      
    terminal --> python predict_house_prices.py

Results:        
California Housing Data:        
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal        
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23        4.526        
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22        3.585        
2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24        3.521        
3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25        3.413        
4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25        3.422        
MeanS quare Error: 0.5558915986952442           
R2 Score: 0.575787706032451     
Model Coefficients:         
Intercept: -37.02327770606416       
coefficients: [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01      
 -2.02962058e-06 -3.52631849e-03 -4.19792487e-01 -4.33708065e-01]       
coefficients for each feature       
            coefficient     
MedInc         0.448675     
HouseAge       0.009724     
AveRooms      -0.123323         
AveBedrms      0.783145     
Population    -0.000002     
AveOccup      -0.003526     
Latitude      -0.419792     
Longitude     -0.433708     

Predicted House Price: $2.54    - $254 000    

This is how you can predict house prices using linear regression that we have used here. And it kind of gives you a better answer.

[⬆ Back to content](#content)


## 61. 9: Weather Forecasting using historical data

[⬆ Back to content](#content)

We will build this weather forecasting model using historical data, where in this project we will predict the temperature or weather conditions using historical weather data.

We'll use linear regression for this simple prediction task.

weather_forecast.py

```python
# import libraries

## handling the data set
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd 

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Get the model and implemented the linear regression algorithm to predict house prices.
from sklearn.linear_model import LinearRegression

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model
## mean_squared_error measures the mean squared error of prediction
## r2_score measures the proportion of variance explained by the model
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
```

Run the file      
    terminal --> python weather_forecast.py

Results:        

Predicted Temperature: 31.70C       

<img src="9 Weather Forecasting using historical data/pics/weather-predict-result.png" width="800" />
<br>
<br>

[⬆ Back to content](#content)


## 62. 10: Basic Neural Network from scratch

[⬆ Back to content](#content)

We will build a basic neural network from scratch in Python, using only numpy to demonstrate how a neural network works under the hood.

We'll implement a simple feedforward neural network with one hidden layer, and use the sigmoid activation function and mean square error as the loss function.

naeural_network.py

```python
# import libraries

## matrix operation and mathematical function
import numpy as np

## Sigmoid activation function and its derivative
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

## Mean Square Error (MSE) Loss Function
## Calculates the mean squared error between the true and the predicted output
## This is the loss function used to measure the performance of the model
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

## Basic Neural Network Class
class BasicNeuralNetwork:
    # Initialize by pass four items which is the object itself input size hidden size and output size which initializes the neural network with random weights and biases
    # weights_input_hidden is the weights between the input layer and the hidden layer
    # weights_hidden_output is the weights between the hidden layer and the output layer
    # bias_hidden is the bias for the hidden layer
    # bias_output - bias for the output layer
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    # To create the forward pass function it takes two parameters - where the input is passed through the network to produce an output
    def foreward(self, X):
        # his will give us the the first hidden input
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output

    # Backward pass and the weights update - where the error is propagated back through the network, and the weights and biases are updated using the gradient descent.
    def backwords(self, X, y, output, learning_rate):
        # This is output error is the difference between the actual output and the predicted output
        output_error = y - output
        # Adjustments for the output layer based on the error and the derivative of the sigmoid function
        output_delta = output_error * sigmoid_derivative(output)
        # This is the error for the hidden layer propagated from the output layer
        # This will give us the hidden error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        # Adjustments for the hidden layer.
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        # Get the weights and biases update which are gradients are used to update the weights and biases using gradient descent
        # So that will give us update the self weights hidden output
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        # this will give us backward pass and weights update
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Train the neural network - where the neural network is trained over multiple epochs to minimize the loss
    def train(self, X, y, epochs, learning_rate):
        # this trains the neural network for a specified number of epochs
        for epoch in range(epochs):
            # Create forward Pass
            output = self.foreward(X)

            # Create the backward pass
            self.backwords(X, y, output, learning_rate)

            # do the loss calculation which is every 100 epochs
            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch: {epoch}, Loss: {loss}")
            
# Training with the XOR data set
## Create XOR data set
## input data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
# target - expected output from the above above calculation
y = np.array([[0],[1],[1],[0]])

# Create a neural network instance with two input neurons two hidden neurons and one output neuron
nn = BasicNeuralNetwork(input_size=2, hidden_size=2, output_size=1)

## Train the neural network
nn.train(X, y, epochs=10000, learning_rate=0.1)

## After training The network's performance is evaluated by comparing predicted and actual outputs
## Test the neural network with the XOR data set.
print("\nTesting the trained neural network:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {nn.foreward(X[i])}, Actual Output: {y[i]}")
```

Run the file      
    terminal --> python naeural_network.py

Results:  

Epoch: 0, Loss: 0.30863473632572136     
Epoch: 100, Loss: 0.24959317442649648       
Epoch: 200, Loss: 0.24951587213632367       
Epoch: 300, Loss: 0.24943111201055101       
Epoch: 400, Loss: 0.24932738644670382       
Epoch: 500, Loss: 0.24919935738995774       
Epoch: 600, Loss: 0.24904004597088747       
Epoch: 700, Loss: 0.24884034893613272       
Epoch: 800, Loss: 0.24858846083864988       
Epoch: 900, Loss: 0.24826922434788817       
Epoch: 1000, Loss: 0.24786346528028563      
Epoch: 1100, Loss: 0.24734740864576782      
Epoch: 1200, Loss: 0.24669229852946356      
Epoch: 1300, Loss: 0.24586432644206113      
Epoch: 1400, Loss: 0.24482488669594502      
Epoch: 1500, Loss: 0.24353105527362978      
Epoch: 1600, Loss: 0.24193615888445474      
Epoch: 1700, Loss: 0.23999055062944633      
Epoch: 1800, Loss: 0.23764332190815973      
Epoch: 1900, Loss: 0.23484642522537524      
Epoch: 2000, Loss: 0.2315629014507764       
Epoch: 2100, Loss: 0.22777965000083533      
Epoch: 2200, Loss: 0.22352187360458617      
Epoch: 2300, Loss: 0.2188621875543082       
Epoch: 2400, Loss: 0.2139162896608085       
Epoch: 2500, Loss: 0.2088226227586159       
Epoch: 2600, Loss: 0.2037129751326756           
Epoch: 2700, Loss: 0.19868603227677306      
Epoch: 2800, Loss: 0.19379210817685816      
Epoch: 2900, Loss: 0.18902934615457728      
Epoch: 3000, Loss: 0.18434642620529895      
Epoch: 3100, Loss: 0.1796461287768625           
Epoch: 3200, Loss: 0.17478592859844788      
Epoch: 3300, Loss: 0.16957428659423895          
Epoch: 3400, Loss: 0.16376499452743806          
Epoch: 3500, Loss: 0.1570595272848032       
Epoch: 3600, Loss: 0.14913939297085005          
Epoch: 3700, Loss: 0.1397524523885711           
Epoch: 3800, Loss: 0.12884044068165512              
Epoch: 3900, Loss: 0.11663889790971757          
Epoch: 4000, Loss: 0.10368375923824649          
Epoch: 4100, Loss: 0.09070477540771826              
Epoch: 4200, Loss: 0.0784291362910215           
Epoch: 4300, Loss: 0.06738869884582632          
Epoch: 4400, Loss: 0.05783647793611309          
Epoch: 4500, Loss: 0.04978460852327064              
Epoch: 4600, Loss: 0.04309769674071763          
Epoch: 4700, Loss: 0.037579002869812                
Epoch: 4800, Loss: 0.03302554816131567          
Epoch: 4900, Loss: 0.029254859515786526             
Epoch: 5000, Loss: 0.02611381753519989          
Epoch: 5100, Loss: 0.02347850014776651          
Epoch: 5200, Loss: 0.021250432466664575         
Epoch: 5300, Loss: 0.019351972156358633         
Epoch: 5400, Loss: 0.017721997239853772             
Epoch: 5500, Loss: 0.016312282129671674         
Epoch: 5600, Loss: 0.015084603009967154         
Epoch: 5700, Loss: 0.014008484673749419         
Epoch: 5800, Loss: 0.013059468575570048         
Epoch: 5900, Loss: 0.012217788128453165             
Epoch: 6000, Loss: 0.011467355885872103         
Epoch: 6100, Loss: 0.01079498729774819              
Epoch: 6200, Loss: 0.010189803292774014         
Epoch: 6300, Loss: 0.009642768073365655         
Epoch: 6400, Loss: 0.009146329415845535             
Epoch: 6500, Loss: 0.008694137000697632         
Epoch: 6600, Loss: 0.008280820441795487         
Epoch: 6700, Loss: 0.00790181324624567          
Epoch: 6800, Loss: 0.007553212321201378             
Epoch: 6900, Loss: 0.007231665158338155         
Epoch: 7000, Loss: 0.006934278700091256         
Epoch: 7100, Loss: 0.006658545293303323         
Epoch: 7200, Loss: 0.006402282189449375             
Epoch: 7300, Loss: 0.0061635818466278525            
Epoch: 7400, Loss: 0.005940770893230264         
Epoch: 7500, Loss: 0.005732376075156898             
Epoch: 7600, Loss: 0.00553709586331205          
Epoch: 7700, Loss: 0.00535377667223484          
Epoch: 7800, Loss: 0.005181392853637158             
Epoch: 7900, Loss: 0.0050190297948909885            
Epoch: 8000, Loss: 0.004865869583039224         
Epoch: 8100, Loss: 0.004721178797905906         
Epoch: 8200, Loss: 0.004584298079570769         
Epoch: 8300, Loss: 0.004454633180571481         
Epoch: 8400, Loss: 0.004331647265322704             
Epoch: 8500, Loss: 0.004214854261168027         
Epoch: 8600, Loss: 0.004103813099355703         
Epoch: 8700, Loss: 0.003998122711714436         
Epoch: 8800, Loss: 0.00389741767119986              
Epoch: 8900, Loss: 0.0038013643828020225            
Epoch: 9000, Loss: 0.0037096577463476407            
Epoch: 9100, Loss: 0.003622018225132257         
Epoch: 9200, Loss: 0.0035381892645759546                
Epoch: 9300, Loss: 0.0034579350136136845            
Epoch: 9400, Loss: 0.0033810383086257667        
Epoch: 9500, Loss: 0.003307298885643709             
Epoch: 9600, Loss: 0.0032365317915380365            
Epoch: 9700, Loss: 0.0031685659690756283            
Epoch: 9800, Loss: 0.003103242994261146         
Epoch: 9900, Loss: 0.003040415947359866         

Since it's 10,000, it stopped at 9900 as the last printed output tested neural output

Testing the trained neural network:     
Input: [0 0], Predicted Output: [[0.05699467]], Actual Output: [0]      
Input: [0 1], Predicted Output: [[0.9475975]], Actual Output: [1]       
Input: [1 0], Predicted Output: [[0.94767365]], Actual Output: [1]      
Input: [1 1], Predicted Output: [[0.05645646]], Actual Output: [0]  

So it did a pretty good job. I would say 99%. Uh good on the prediction. So there you go. We have created our own basic neural network from scratch.

[⬆ Back to content](#content)


## 63. 11: Stock Price Prediction using historical data with simple Linear Regression

[⬆ Back to content](#content)

Let's implement stroke price prediction using historical data with linear regression.

We will use past stock prices to predict the future price of a stock.

This project will involve using a data set that contains dates and closing prices, and we will use linear regression to predict the future stock prices.

stock_price_prediction.py

```python
# inport libraries

## Pandas is used for data manipulation and handling time series data.
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Get the model and implemented the linear regression algorithm to predict house prices.
from sklearn.linear_model import LinearRegression

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model
## mean_squared_error measures the mean squared error of prediction
## r2_score measures the proportion of variance explained by the model
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
```

Run the file        
    terminal --> python stock_price_prediction.py       

Result:     

     Date  Close        
0  738886    150        
1  738887    152        
2  738888    153        
3  738889    155        
4  738890    154        

Mean Squared Error: 0.06023335316288063     

R2 Error: 0.9950829915785404        


<img src="11 Stock Price Prediction with Linear Regression/pics/stock-price-prediction-result.png" width="800" />
<br>
<br>

Predicted Stock Price: $161.21


[⬆ Back to content](#content)


## 64. 12: Predict Diabetes using logistic regression

[⬆ Back to content](#content)

Let's implement a diabetes prediction using logistic regression.

We will use the Pima Indians Diabetic data set, which is a popular data set for predicting whether a patient has diabetes based on various medical features such as glucose level, blood pressure, BMI, and others.

diabetes_prediction.py

```python
# import libraries

## Pandas is used for data manipulation and handling time series data.
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Binary classification problems in which in our case is spam versus not spam
from sklearn.linear_model import LogisticRegression

# Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification - displays true false positives and negatives.
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class - detailed performance report of a model.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## data visualization - create a heat map from the confusion matrix visualization
import seaborn as sns

## Plotting the actual predicted stock prices
import matplotlib.pyplot as plt

# Load the Pima Indians Diabetes data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPredigreeFunction', 'Age', 'Outcome']

## Load the dataset
df = pd.read_csv(url, names=column_names)

## Display first few rows of the dataset to see data structure
print("Diabetes Dataset:")
print(df.head())

## Check the dataset structure terminal --> python diabetes_prediction.py

# Get the features. We want all the columns except for outcomes and features
## Features
X = df.drop('Outcome', axis=1)
## Target
y = df['Outcome']

## Splits data into training and testing sets
## Split the data set using the train test split, which splits the data set into 80% for training and 20% for testing, and also will give a random state for it, which is one second
## X and y are features and my target
## test_size=0.2 mean that 20% of the data size will be used for testing and 80% of the data will be used as a training data
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
## This will give us variables - training data, testing data, training targets and testing targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
## Get hte model
model = LogisticRegression(max_iter=1000)

## Train the model
model.fit(X_train, y_train)

## Make prediction
y_pred = model.predict(X_test)

## Evaluate the model based on this prediction using accuracy, confusion matrix and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

## Check the results of the evaluation
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: {conf_matrix}")
print(f"Classification Report: {class_report}")

## Visualize the confusion matrix I can use the seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

## Test the model with the new data set
new_data = pd.DataFrame({
    'Pregnancies': [5],
    'Glucose': [120],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [80],
    'BMI': [32.0],
    'DiabetesPredigreeFunction': [0.5],
    'Age': [42]
})

## Predicted outcome
predicted_outcome = model.predict(new_data)

## Print the result of the new data
print(f"Predicted Outcome: {'Diabetic' if predicted_outcome[0] == 1 else 'Non-Diabetic'}")
```

Run the file        
    terminal --> python diabetes_prediction.py      

Result:  

Diabetes Dataset:       
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPredigreeFunction  Age  Outcome       
0            6      148             72             35        0  33.6                      0.627   50        1       
1            1       85             66             29        0  26.6                      0.351   31        0       
2            8      183             64              0        0  23.3                      0.672   32        1       
3            1       89             66             23       94  28.1                      0.167   21        0       
4            0      137             40             35      168  43.1                      2.288   33        1       
Accuracy: 0.7467532467532467        
Confusion Matrix: [[78 21]      
 [18 37]]       
Classification Report:               precision    recall  f1-score   support        

           0       0.81      0.79      0.80        99       
           1       0.64      0.67      0.65        55       

    accuracy                           0.75       154       
   macro avg       0.73      0.73      0.73       154       
weighted avg       0.75      0.75      0.75       154       


<img src="12 Predict Diabetes using logistic regression/pics/diabetes-results.png" width="600" />
<br>
<br>

Predicted Outcome: Non-Diabetic

[⬆ Back to content](#content)


## 65. 13: Dog vs. Cat Classifier with CNN

[⬆ Back to content](#content)

We will build a dog versus cat classifier using a convolutional neural network, which is CNN. This project involves classifying images of dogs and cats into their respective categories using deep learning.

We will use CNN for feature extraction and classification in this case. So for this example, we'll use the Kaggle Dogs versus cats dataset, which contains images of dogs and cats.

For demonstration purposes we will outline how you can structure and implement this model.

The dataset can be downloaded from Kaggle - https://www.kaggle.com/datasets/tongpython/cat-and-dog

Make sure that the data sets are with the structure as follow:

data/
├── training_set/
│   ├── cats/        ← subfolder for each class
│   │   ├── cat1.jpg
│   │   └── ...
│   └── dogs/        ← subfolder for each class
│       ├── dog1.jpg
│       └── ...
└── test_set/
    ├── cats/
    └── dogs/
dogs_vs_cats.py
test-image.jpg

Process steps:
1. Training the model and save it
2. Evaluate the model
3. Test with a particular image to see if it classifies properly

Once the model is trained and saved, the same model it will be used every time we run the file dogs_vs_cats.py.
If the model is wrongly trained we need to delete it and then train it again with appropriate data.

dogs_vs_cats.py

```python
# import libraries
import os

## matrix operation and mathematical function
import numpy as np

## TensorFlow is a deep learning library
## TensorFlow provides the tools for building and training neural networks
import tensorflow as tf 

## TensorFlow Keras is Keras which is included with TensorFlow simplifies the neural network creation. 
## We have layers which is used to add layers to our convolutional neural network (CNN)
## We have models which use to define and compile a neural network model
from tensorflow.keras import layers, models

## Image data generator is a class in Keras used for image data augmentation which improves the models ability to generalize by creating variations in the training data.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## We are going to use custom image
from tensorflow.keras.preprocessing import image

## Used for plotting the training and validation accuracy loss curves
import matplotlib.pyplot as plt

## Define paths to the data set and update them with the actual dataset location
## Dataset url - https://www.kaggle.com/datasets/tongpython/cat-and-dog
train_dir = 'data/training_set'
validation_dir = 'data/test_set'

## Define the iImageDataGenerators for data augmentation and rescaling
## Train data gen prepares the training data by applying data augmentation techniques such as rescaling rotation, zoom and horizontal flipping, so it helps the model generalize better by artificially creating new training examples.
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Rescale pixel values from (0-255) tp (0-1)
    rotation_range=40,              # Randomly rotate images for get better testing
    width_shift_range=0.2,          # Randomly shift images horizontally
    height_shift_range=0.2,         # Randomly shift image vertically
    shear_range=0.2,                # Randomly shear images
    zoom_range=0.2,                 # Randomly zoom in on images
    horizontal_flip=True,           # Randoly flip images horizontally
    fill_mode='nearest'             # Fill pixels that may have been lost after tranformation
)

## For the validation data we rescale. We don't have to do any data augmentation.
validation_datagen = ImageDataGenerator(rescale=1./255)

## Each sub directory within this train directory or validation directory should correspond to a particular class which is cats and dogs.
## Loading the training data and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),         # Resize all iages to 150x150
    batch_size=32,
    class_mode='binary',            # Binary classification (Dog or Cat)
)

## Create the validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),         # Resize all iages to 150x150   
    batch_size=32,
    class_mode='binary',            # Binary classification (Dog or Cat)
)

# Define a path to save the model
MODEL_PATH = 'cat_dog_model.keras'

# Check if a saved model exists
if os.path.exists(MODEL_PATH):
    # Load the existing model instead of training again
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    # Define the CNN model
    model = models.Sequential()
    
    # Create the first convolutional model/layer
    ## Conv2D creates convolutional layers to extract spatial features from the images
    ## First ayer has 32 filters and size 3 by 3, which is the kernel size for all the layers and the activation style
    ## We are using ReLU activation and input shape. Since we are converting that 150 by 150, we are specifying 150 by 150
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Create the second convolutional model/layer
    ## The second layer has 64 filters
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Create the third convolutional model/layer
    ## The third layer has 128 filters
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Create the third convolutional model/layer
    ## The third layer has 128 filters
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    ## Flatten the output from the convolutional layer and add fully connected layers.
    model.add(layers.Flatten())
    ## The first dense layer has 512 units with ReLU activation.
    model.add(layers.Dense(512, activation='relu'))
    ## Final Dense layer which has only one unit with sigmoid activation to predict a binary outcome, which is cat or dog
    model.add(layers.Dense(1, activation='sigmoid')) # Output layer for binary classification

    # Compile the model
    ## Adam optimizer is used for efficient gradient based optimization
    ## loss='binary_crossentropy' function is suitable for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ## Print summary of the model
    model.summary()

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=50
    )
    
    ## Plot the training and validation accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    ## Training accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    ## Training loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

    # Save the model after training
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
## Test the model with the custom image
## This loads and pre-process a custom image and then makes a prediction using the trained model
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Load the image
    img_array = image.img_to_array(img)                 # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)       # Add batch dimension
    img_array /= 255.0                                  # Normalize the image (rescale pixel values to [0, 1])

    prediction = model.predict(img_array)               # Make the prediction

    if prediction[0] > 0.5:
        print(f"The image is predicted to be a Dog with a confidence of {prediction[0][0]:.2f}")
    else:
        print(f"The image is predicted to be a Cat with a confidence of {1 - prediction[0][0]:.2f}")

# Example: Test the classifier with a new image
predict_image(model, 'test-image-dog.jpg')
predict_image(model, 'test-image-cat.jpg')
```

Donload image of a cat or a dog and add it to the prediction test above.

Run the file        
    terminal --> python dogs_vs_cats.py   

Result:         

<img src="13 Dog vs. Cat Classifier with CNN/pics/cat_dog_result2.png" width="800" />
<br>
<br>      

<img src="13 Dog vs. Cat Classifier with CNN/pics/cat_dog_result.png" width="1000" />
<br>
<br>

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 119ms/step
The image is predicted to be a Dog with a confidence of 0.86
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step
The image is predicted to be a Cat with a confidence of 0.83

[⬆ Back to content](#content)


## 66. 14: Tic-Tac-Toe AI using Minimax Algorithm

[⬆ Back to content](#content)

We will build a tic tac toe AI using the minimax algorithm.

The minimax algorithm is a recursive approach used for making decisions in turn based games like tic tac toe.

The AI will simulate all possible moves, evaluate them, and choose the best one to either maximize its own chance of winning or minimize the player's chance of winning.



```python
# Define a board as a list
## The tic tac toe board is represented as 1D list of nine elements. Index is 0 to 8. Each element starts as an empty space with double quotes or quotes spaced to indicate that the slot is available for a move.
board = [' ' for _ in range(9)] # 3x3 Tic_tac_toe board represented as a 1D list

# Function to print the tic tac toe board.
## This function prints the board in a user friendly format. It divides the 1D list into three rows, each containing three elements, and prints them as rows separated by vertical bars to mimic a tic tac tic tac toe board. So that's why here from I'm going from each element and printing those pipes to separate them as elements.
def print_board(board):
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print('| ' + ' | '.join(row) + ' |')

# Function to check if there is a winner
def check_winner(board, player):
    ## Those are our win conditions
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

# Function to check if the board is full
def is_board_full(board):
    return ' ' not in board

# Function to evaluate the board for minimax algorithm
## This evaluate function assigns a score based on the current board state one is the one. If the I which is zero, wins minus one if the human player which is X wins, and zero if the game is still ongoing and ends in a draw.
def evaluate(board):
    if check_winner(board, 'O'):    # AI is 'O'
        return 1
    elif check_winner(board, 'X'):  # Human is 'X'
        return -1
    else:
        return 0

# Minimax function to calculate the best move for the AI.
def minimax(board, depth, is_maximizing):
    # evaluate the board
    score = evaluate(board)
    # if score is one or score equal to minus one or if score board is full like the board is full then we'll return the score
    if score == 1 or score == -1 or is_board_full(board):
        return score
    
    if is_maximizing:   # AI's turn
        best_score = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'  # AI makes the move
                best_score = max(best_score, minimax(board, depth + 1, False))
                board[i] = ' '  # Undo move
        return best_score

    else:       # Human move
        best_score = float('inf') 
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'  # Humam make move
                best_score = min(best_score, minimax(board, depth + 1, True))
                board[i] = ' '  # Undo move
        return best_score

# Function to find the best move for the AI
## This function loops over all the available positions on the board, simulates the I making a move at each position and uses the minimax algorithm to evaluate the move. It chooses the move with the highest value. Best move for the I in this case.
def find_best_move(board):
    best_value = -float('inf')
    best_move = -1

    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'     # AI makes move
            move_value = minimax(board, 0, False)
            board[i] = ' '     # Undo move)
            
            if move_value > best_value:
                best_value = move_value
                best_move = i
        
    return best_move

# Main game loop
## This is the main game loop where the human player enters their move by providing a number between 1 and 9 to correspond with the board positions. If the position is valid that is empty, the player makes their move X and the game checks if the player has won.
def play_game():
    while True:
        print_board(board)

        # Player move
        player_move = int(input("Enter your move (1-9): ")) - 1
        if board[player_move] != ' ':
            print("Invalid move! Try Again.")
        board[player_move] = 'X'

        # Check if player won
        if check_winner(board, 'X'):
            print_board[board]
            print("You win!")
            break

        # Check for a draw
        if is_board_full(board):
            print("Its a draw")
            break

        # AI move
        ai_move = find_best_move(board)
        board[ai_move] = 'O'

        # Check if AI won
        if check_winner(board, 'O'):
            print_board(board)
            print("AI wins!")
            break
            
        # Check for draw
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break

# Start the game
play_game()
```

Play the game        
    terminal --> python tic-tac-toe-minimax-alg.py   


[⬆ Back to content](#content)


## 67. 15: Credit Card Fraud Detection using Scikit-learn

[⬆ Back to content](#content)

In this project, we will build a credit card fraud detection model using the scikit learn. The goal is to classify credit card transactions as fraudulent or non fraudulent based on features like transaction amount, time and other numerical data.

We will train a classification model using a common data set for fraud detection, and evaluate it based on accuracy and precision.

For this particular project, I have downloaded Kaggle's Credit card fraud detection data set, which link I have provided in the description.

Download the dataset - https://www.kaggle.com/code/gaamoucimohamed/creadit-card-fraud-detection/input

Set the dataset in folder data/creaditcard.csv

creadit_card_fraud_detection.py

```python
# inport libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## This scales features to have a mean of zero and standard deviation of one helping improve model stability.
from sklearn.preprocessing import StandardScaler

## Robust classifier that works well for binary classification tasks
from sklearn.ensemble import RandomForestClassifier

# Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('data/creditcard.csv')

# Display first few rows from dataset sample
print("Dataset Sample:\n", df.head())

# Separated features X and target Y
X = df.drop('Class', axis=1)        # 'Class' column is the target, with 0 for non-fraud
y = df['Class']

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features for numerical stability
## StandardScaler() - standardizes features to improve numerical stability
scaler = StandardScaler()

## The fit transform fits the scalar on x train and transforms it
X_train = scaler.fit_transform(X_train)

## Transforms the x underscore test using the scaling parameters from x underscore train
X_test = scaler.transform(X_test)

# Initialize and train the model 
## Initialize the model with 100 trees
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
model = RandomForestClassifier(n_estimators=100, random_state=42)

## Train the model on the training data that we have
model.fit(X_train, y_train)

## Make prediction on the test set
## this function uses the trained model model to predict labels for X_test.
y_pred = model.predict(X_test)

# Evaluate the model
## This calculates the overall accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
## This provides precision recall and F1 score for each class fraud and non fraud
report = classification_report(y_test, y_pred)
## Generates a confusion matrix showing true positives, false positives, true negatives and false negatives. So We'll get an matrix of these four elements.
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)
```

Run the file                
    terminal --> python creadit_card_fraud_detection.py     

Results:

<img src="15 Credit Card Fraud Detection using Scikit-learn/pics/creadit_card_fraud_detection.png" width="800" />
<br>
<br>

[⬆ Back to content](#content)


## 68. 16: Iris Flower Classification using decision trees

[⬆ Back to content](#content)

Let's build a project to classify the iris flower dataset using decision trees. The Iris dataset is a well known dataset in machine learning, containing data about the sepal and petal length and width of three types of iris flowers setosa, versicolor and virginica.

We'll train a decision tree classifier to predict the species of the iris flower based on these measurements. If you have never heard of this, I would say just do some research and come back and you'll understand what I'm talking about.

iris_flower_classification.py

```python
# inport libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Load data
from sklearn.datasets import load_iris

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Implements the decision tree classifier from scikit learn
from sklearn.tree import DecisionTreeClassifier

## Import metrics from scikit learn
from sklearn import metrics

## Function to visualize the trained decision tree
from sklearn.tree import plot_tree

## JUsed to visualize the decision tree
import matplotlib.pyplot as plt

# Load the load and display the iris data set
## Load the iris data set which includes features like sepal length, sepal width, petal length and petal width
iris = load_iris()

## Create a DataFrame from the Iris dataset
## Converts the data set into pandas data frame for easier manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

## Add the species column (target)
df['species'] = iris.target

## Display the first few rows of the data set
print("Iris Dataset")
print(df.head())

## split the dataset into features (X) and target (y)
## The features include sepal length, sepal width, petal length and petal width
X = df.drop('species', axis=1)

## Target variable
## A species is either setosa versicolor or virginica
y = df['species']

## Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 30% of the data to be test, whereas 70% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
## Initialize classifier
classifier = DecisionTreeClassifier()

## Trains the classifier on the training set
classifier.fit(X_train, y_train)

## Make predictions on the test set - Predicts the species of the iris flowers in the test set
y_pred = classifier.predict(X_test)

# Evaluate the model using accuracy, confusion matrix and classification report
## accuracy score calculates the accuracy of the model, that is the percentage of correct predictions
accuracy = metrics.accuracy_score(y_test, y_pred)
## It generates the confusion matrix, which shows the number of correct and incorrect predictions for each class
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
## Classification report provides a detailed classification report including precision recall and F1 score for each class
class_report = metrics.classification_report(y_test, y_pred)

## Print results
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix")
print(conf_matrix)
print("\Classification Report")
print(class_report)

# Visualize the decision Tree
plt.figure(figsize=(12,6))
## So what feature_names=iris.feature_names does is is the labels the features which is the sepal width
## class_names=iris.target_names labels the target classes setosa, versicolor and virginica, and filled true colors the tree nodes based on the class they predict, providing a visual understanding of the decision process
## filled=True - filled true colors the tree nodes based on the class they predict, providing a visual understanding of the decision process
plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
## Set plot title
plt.title("Decision Tree for Iris Flower Classification")
## Print the plot
plt.show()

```

Run the file                
    terminal --> python iris_flower_classification.py 

Results:

<img src="16 Iris Flower Classification using decision trees/pics/decision_tree_iris_flowers.png" width="800" />
<br>
<br>

<img src="16 Iris Flower Classification using decision trees/pics/Iris_flowers_result_data.png" width="600" />
<br>
<br>


[⬆ Back to content](#content)


## 69. 17: Simple Personal Assistant using Python speech libraries

[⬆ Back to content](#content)

Let's build a simple personal assistant today in Python that can listen to voice commands. Perform basic tasks like searching the web, checking the time or greeting the user, and respond with audio feedback.

We'll use the speech recognition library to recognize speech and the python, which is pi to three library to synthesize speech for responses.

Create a separate virutal envronment and activate it     
    terminal --> python -m venv speechenv       
    terminal --> speechenv/scripts/activate

Install Libraries
    terminal --> pip install SpeechRecognition pyttsx3

personal_assistant_python.py

```python
# Install libraries
## pip install SpeechRecognition pyttsx3 pyaudio

# import libraries
## speech revognition
import speech_recognition as sr

## speech a text to speech allowing the assistant to respond vocally
import pyttsx3

## Provides date and time information
import datetime

## Using web browser
import webbrowser

## Allows the assistant to run system commands like opening notepad or calculator or any other tools that you have
import os

# Initialize the speech engine for text to speech
engine = pyttsx3.init()

# Function to make the assistant speak
def speak(text):
    print(f"Assistant: {text}")  # add this to debug
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

# Function to take a voice command from the user
## This command this function kind of listens to the user's voice using the microphone, converts the audio to text using Google speech recognition, and returns the text command in lowercase back to us
def take_command():
    # This will recognize the voice
    recognizer = sr.Recognizer()        
    with sr.Microphone() as source:
        print("Listening...")
        # Adjust for background noise to improve recognition
        recognizer.adjust_for_ambient_noise(source)
        # Activelly listening
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            # Call that function Google API with the audio and see if we can understand what it's saying
            command = recognizer.recognize_google(audio)
            print(f"User said: {command}")
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Network error.")
            return None

    # Command it converted from speech to text will be returned back
    return command.lower()

# Function to respond to different commands
## This function handles the assistance responses based on the user's command
def respond(command):
    if 'hello' in command or 'hi' in command:
        speak("Hello! How can I help you today?")

    elif 'time' in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")

    elif 'weather' in command:
        speak("Searching for current weather")
        webbrowser.open("https://www.google.com/search?q=current+weather")

    elif 'search' in command:
        search_query = command.replace('search for', '').replace('search', '').strip()
        if search_query:
            speak(f"Searching for {search_query}")
            webbrowser.open(f"https://www.google.com/search?q={search_query}")

    elif 'firefox' in command:
        speak("Opening Firefox")
        os.system("start firefox")

    elif 'calculator' in command:
        speak("Opening Calculator")
        os.system("start calc")

    elif 'bye' in command or 'exit' in command or 'quit' in command:
        speak("Goodbye! Have a great day.")
        exit()

    else:
        speak("I'm sorry, I don't know that command.")

# Main function to run the assistant
def run_assistant():
    speak("Hello I am your assistent. How can I help you?")
    while True:
        command = take_command()
        if command:
            respond(command)

# Start the assistant
run_assistant()
```

Run the file                
    terminal --> python personal_assistant_python.py

Results:

[⬆ Back to content](#content)


## 70. 18: Text Summarizer using Gensim

[⬆ Back to content](#content)

create a text summarizer using NLTK, where this approach will use frequency based extractive summarization, where we calculate the frequency of each word in the text and score each sentence based on the sum of the frequencies of the words it contains. Sentences with higher scores are considered more important and selected for the summary.

text_summarizer.py

```python
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

```

Run the file                
    terminal --> python text_summarizer.py

Results:

```text
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal.
A subset of artificial intelligence is machine learning, which refers to the concept that computer programs can automatically learn from and adapt to new data without being explicitly programmed.
Deep learning techniques enable this automatic learning through the absorption of huge amounts of unstructured data such as text, images, or video.


Summary:
 A subset of artificial intelligence is machine learning, which refers to the concept that computer programs can automatically learn from and adapt to new data without being explicitly programmed. 
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
```

[⬆ Back to content](#content)


## 71. 19: Fake Product Review Detection using NLP techniques

[⬆ Back to content](#content)

We will create a fake product review detection model using NLP, which is natural language processing techniques.

The goal is to identify whether a given product review is genuine or fake based on its text content. We will pre-process the reviews, vectorize the text using TFIDF, and then train a classifier to predict if a review is fake.

Download the dataset - https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

Set it in folder data/fake_reviews_dataset.csv

Create the file fake_product_review_detection.py

```python
# import libraries
## handling the data set
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd 

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Converts text into numerical features by calculating term frequency. Inverse document frequency.
from sklearn.feature_extraction.text import TfidfVectorizer

## Simple classifier to predict whether a review is fake or genuine
from sklearn.linear_model import LogisticRegression

# Metrics to evaluate the model's performance
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class
from sklearn.metrics import  accuracy_score, classification_report

# Load the dataset (replace 'review.csv' with the actual dataset path)
df = pd.read_csv('data/fake_reviews_dataset.csv')
df = df.rename(columns={'text_': 'review_text'})
df['label'] = df['label'].map({'OR': 'genuine', 'CG': 'fake'})

# Display the first few rolls of the dataset
print("Dataset Seample\n", df.head())

# Define features (X) and target (y)
X = df['review_text']   # Assuming the dataset has a 'revirew_text' column
y = df['label']     # Assuming the dataset has 'label' column with values 'fake' or 'genuine'

# Splitting the data set into training and testing data testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Tfidfvectorizer
## stop_words='english' removes common English stopwords to focus on important words, it will remove words like 'and', 'if', 'that', 'the' 'all those' and easy words which are used quite a lot
## max_df=0.7 ignores words that appear in more than 70% of the documents. If there is something that comes quite a lot like food, food, food, since it's a review, it might have that review text over there.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
## This learns vocabulary from Xtrain and transforms it into TF-IDF vector
X_train_tfidf = vectorizer.fit_transform(X_train)   # Fit and transform the training
## This transforms X test to TF-IDF vectors using the vocabulary from Xtrain
X_test_tfidf = vectorizer.transform(X_test)         # Only transfor the test data

# Initialize the Logistic Regression Classifier
## This logistic regression initializes the logistic regression classifier suitable for binary classification task in our case
model = LogisticRegression()

# Train the model
## This trains the logistic regression model on the training data. Learning the relationship between TF-IDF vectors and labels.
model.fit(X_train_tfidf, y_train)

# Make the predictions on the test set
## It uses the trained model to predict labels for test data. Identifying whether reviews are fake or genuine.
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
## Calculates the overall accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
## Generates a detailed report with metrics like precision, recall and F1 score for each class. Fake and genuine.
report = classification_report(y_test, y_pred)

# Print results
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report\n", report)

```

Run the file                
    terminal --> python fake_product_review_detection.py

Results:

<img src="19 Fake Product Review Detection using NLP techniques/pics/fake_review_detection_result.png" width="600" />
<br>
<br>


[⬆ Back to content](#content)


## 72. 20: Detect Emotion in Text using Natural Language Toolkit (NLTK)

[⬆ Back to content](#content)

We will create a project to detect emotion in text using the Natural Language Toolkit, where we'll analyze the sentiment of a given text to determine if it expresses emotions like joy, sadness, anger, etc. this example will focus on basic sentiment analysis to determine if a text is positive, negative, or neutral.


emotion_detection.py

```python
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

```

Run the file                
    terminal --> python emotion_detection.py

Results:

<img src="20 Detect Emotion in Text using Natural Language Toolkit (NLTK)/pics/emotion_detection_result.png" width="600" />
<br>
<br>


[⬆ Back to content](#content)


## 73. 21: Book Recommendation System using collaborative filtering

[⬆ Back to content](#content)

We will create a book recommendation system today using the collaborative filtering, where collaborative filtering is a popular technique used in recommendation systems where recommendations are based on user item interactions.

Here we'll use the item based collaborative filtering approach to recommend books based on user preferences.
We'll use the cosine similarity measure to find similar books based on user's rating.

For this example, we'll use a sample data set of books ratings by users.

book_recommendation.py

```python
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
```

Run the file                
    terminal --> python book_recommendation.py

Results:

<img src="21 Book Recommendation System using collaborative filtering/pics/books_recommended_resutl.png" width="400" />
<br>
<br>



[⬆ Back to content](#content)

## 74. 22: Predict Car Prices using Random Forest

[⬆ Back to content](#content)

We will create a car prediction car price prediction model using Random Forest. The goal is to predict the price of a car based on various features like make, model, year, mileage, etc..

Now, Random Forest is an ensemble learning method that is suitable for regression task as it combines multiple decision trees to improve prediction accuracy.

predict_car_prices.py

```python
# import libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Implements the Random Forest Regressor model for predicting prices
from sklearn.ensemble import RandomForestRegressor

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model.
## mean_squared_error measures the mean squared error of prediction
## r2_score measures the proportion of variance explained by the model
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
```

Run the file                
    terminal --> python predict_car_prices.py

Results:

Mean Squared Error: 23447450.0          # since the number is so big, it might not be as good
R-squared Score: 0.22487768595041324    # since it is not close to one is like 0.22 tt might be not perfect.

The reason it's not as good is because my data set contains ten items, and if you provide more data to it, it will work better.


[⬆ Back to content](#content)

## 75. 23: Identify Fake News using Naive Bayes

[⬆ Back to content](#content)

We will be building a fake news detection model using Naive Bayes with a large data set.

We will use the fake news dataset from Kaggle. - https://www.kaggle.com/datasets/algord/fake-news

Dowload and set it in data/FakeNewsNet.csv

This one includes labels for fake and real news articles by calling it real. If the value is one, it's real. If it's not, it's zero.

fake_news_identifier.py

```python
# import libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Converts text data into numerical features using term. In the CSV data file we text information. We want to convert/vectorize it.
from sklearn.feature_extraction.text import TfidfVectorizer

## Naive Bayes classifier suitable for text data
from sklearn.naive_bayes import MultinomialNB

# Metrics to evaluate the model's performance
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class
from sklearn.metrics import  accuracy_score, classification_report

# Load the dataset
## https://www.kaggle.com/datasets/algord/fake-news
## If you have different dataset, replace the FakeNewsNet.csv with the actual one.
## The dataset should have a text column containing 'news' and 'content' and a label column which with values like 'fake' or 'real' or similar to that we have. Or you can change it accordingly.
df = pd.read_csv('data/FakeNewsNet.csv')

# Display the first few rows of the dataset
print("Dataset Sample:\n", df.head())

# Define features (X) and target (y)
X = df['title']   # Assuming the dataset has a 'text' column for the news content
y = df['real']    # Assuming the dataset has a 'label' column with values 'fake' or 'real'

# Split the dataset into training and testing sets with stratification
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes are proportionally represented in the training and test sets whether it's fake or real news
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the TfidfVectorizer to convert text into numerical features
## Converts text data into TF-IDF scores. The stop_words='english' remote's common English stopwords to focus on important words we don't want 'and', 'the', 'if', all those words
## max_df=0.7 ignores words appearing in more than 70% of document, as they are likely to common to be useful. For example, if there are words like 'news' or 'channel' we don't need them because they will be everywhere.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
## fit_transform(X_train) learns the vocabulary from X_train data and transforms it into TF-IDF vectors
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform the training data
## vectorizer.transform(X_test)  converts X_test data into tf TF-IDF vectors using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)        # Only transform the test data

# Initialize the Naive Bayes classifier
## Initializes the Naive Bayes classifier which is effective for text classification task where features like words are multinomial distributed.
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
## This uses the trained model to classify the articles in the test set X_test_tfidf. The predicted labels y_pred indicate whether each article is fake or real.
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
## This calculates the accuracy of the model, that is, the percentage of correctly classified articles in the test set.
accuracy = accuracy_score(y_test, y_pred)
## Generates a detailed report including precision recall and F1 score for each class whether it is fake or real. 
## zero_division=1 parameter avoids warnings for any labels without true samples by setting the undefined metrics to 1.0.
report = classification_report(y_test, y_pred, zero_division=1)

## Print the accuracy and the classification report. So this will give us a good idea about how good the model is.
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)

```

Run the file                
    terminal --> python fake_news_identifier.py

Results:

<img src="23 Identify Fake News using Naive Bayes/pics/fake-news-result.png" width="1000" />
<br>
<br>



[⬆ Back to content](#content)


## 76. 24: Create a Resume Scanner using keyword 

[⬆ Back to content](#content)

We will create a resume scanner that uses keyword extraction to analyze resumes and determine if candidates have the required skills for a job. This scanner will take a resume's text and extract relevant keywords allowing recruiters to quickly match candidates to the job requirements.

We will use TF-IDF which is term frequency inverse document frequency to identify keywords from the resume text and match them against predefined job keywords.

resume_scanner.py

```python
# import libraries

## Used for loading and manipulating data sets
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Converts text data into numerical features using term
from sklearn.feature_extraction.text import TfidfVectorizer

## Calculates the cosine similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity

# Sample resumes and job description data
## Each resume has a unique resume ID and resume text detailing the candidate's skills. Now we can do this, or we can use one of the tools to read a resume from a PDF format or a doc format and then create a data structure. But we are directly taking this from from a dictionary where I have these resume IDs. For those resume IDs we have resume text.
data = {
    'resume_id': [1, 2, 3],
    'resume_text': [
        "Experienced data scientist with skills in Python, machine learning, and data analysis.",
        "Software developer with expertise in Java, cloud computing, and project management.",
        "Data analyst with proficiency in SQL, Python, and data visualization."
    ]
}

job_description = "Looking for a data scientinst skilled in Puthon, machine learning, SQL, and data analysis."

# Convert to DataFrame
## Convert the dictionary into a DataFrame DF making it easier to analyze and manipulate.
df = pd.DataFrame(data)
## print the resumes just to look at how the data frame looks like
print("Resumes:\n", df)

# Combine job description with resumes for TF-IDF vectorization
## Convert the resume text column to a list of strings which is documents
documents = df['resume_text'].tolist()
## Append the job description which adds the job description to the list, allowing us to compute the TF-IDF for both resumes and the job description.
documents.append(job_description)

# Initialize the TfidfVectorizer
## This initializes the Vectorizer and removes common English stop words like 'and', 'the', 'off' etc.
vectorizer = TfidfVectorizer(stop_words='english')
## Fit the Vectorizer on the documents list and transforms each document into a TF-IDF vector, creating a TF-IDF matrix.
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarity scores between job description and each resume
## Calculates the similarity between job description, which is large document in tf IDF matrix and each resume in all the documents except the last one
## flatten() converts the results to a 1D array, making it easier to handle
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Display similarity scores for each resume
## adds the similarity scores as a new column in the data frame, associating each resume with its similarity to the job description
df['similarity_score'] = similarity_scores
## Print the similarity score
print("\nResume Similarity Scores:\n", df[['resume_id', 'similarity_score']])

# Identify resumes that match the job requirements (threshold can be adjusted)
## threshold = 0.2 sets the threshold similarity score to consider resume as a match for the job requirements
threshold = 0.2
## Filters resumes with similarity score above the threshold.
matching_resumes = df[df['similarity_score'] >= threshold]
## print out - saying which will display IDs and similarity scores for the resumes that meet the threshold where I'm giving, getting the resume ID and the similarity score so we can check it out.
print("\nResumes matching the job requirements:\n", matching_resumes[['resume_id', 'similarity_score']])

```

Run the file                
    terminal --> python resume_scanner.py

Results:

<img src="24 Create a Resume Scanner using keyword/pics/resule-scanner-results.png" width="600" />
<br>
<br>


[⬆ Back to content](#content)


## 77. 25: Customer Churn Prediction using classification algorithms

[⬆ Back to content](#content)

we'll build a customer churn prediction model using a classification algorithm. In this case, we'll use the logistic regression. 

The goal is to predict whether a customer will churn, which means leave the company based on various features.

We'll train the model using a dataset with customer data such as tenure, usage and demographic information.

customer_prediction.py

```python
# import libraries

## pandas used for loading and manipulating the data set
## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## This scales features to have a mean of zero and standard deviation of one helping improve model stability.
from sklearn.preprocessing import StandardScaler

## Binary classification problems in which in our case we are going to predict churn or no churn
from sklearn.linear_model import LogisticRegression

# Metrics to evaluate the model's performance
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class
from sklearn.metrics import  accuracy_score, classification_report

# Load the dataset
## replace the customer churn to any other data that you should work with. The data should have data set should have various features related to customer information and the churn column indicating churn status. You need to have that in your data set.
df = pd.read_csv('data/customer_churn.csv')

# Display the first few rows of the dataset
## This displays the first few rows of the data set to confirm it's loaded correctly
print("Dataset Sample:\n", df.head())

# Basic data preprocessing: handle missing values (if any)
## Removes any rows with missing values to ensure data integrity
df = df.dropna()

# Convert categorical columns to numeric using one-hot encoding
## If it's not numerical it will convert into numerical.
df_encoded = pd.get_dummies(df, columns=['gender', 'contract_type', 'payment_method'])

# Define features (X) and target (y)
## Drop the churn in along axis one. So that will get us the features.
X = df_encoded.drop('churn', axis=1)  # Features
## Use churn as target variable
y = df_encoded['churn']               # Target variable

# Split the dataset into training and testing sets
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes (churn and non churn) are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling for numerical stability
## Standardizes features to have a mean of zero and a standard deviation of one.
scaler = StandardScaler()
## Calculate scaling parameters Metres mean and standard deviation on the training set and applies proper scaling.
X_train = scaler.fit_transform(X_train)
## Appliy the same scaling parameters from X_train to the test set (X_test) for consistency.
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression classifier
## This initializes the logistic regression classifier, which is commonly used for binary classification tasks like churn prediction.
model = LogisticRegression()

# Train the model
## The fit function trains the logistic regression model on the training data of X_train and y_train, learning the relationship between the features and the churn status.
model.fit(X_train, y_train)

# Make predictions on the test set
## This uses the trained model to predict the churn status for customers in the test set which is X_test, the predictions y_pred indicate whether each customer is classified as having churn or not.
y_pred = model.predict(X_test)

# Evaluate the model
## This calculates the accuracy of the model, which is the percentage of correctly classified customers in the test set
accuracy = accuracy_score(y_test, y_pred)
## Generates a report including precision recall and F1 score for each class.
report = classification_report(y_test, y_pred, zero_division=1)

# Print results
## So this will print out accuracy, which will show the overall percentage of correctly classified samples.
print(f"\nAccuracy: {accuracy}")
## And the classification report will show us the metrics like precision recall and F1 score for each class.
print("\nClassification Report:\n", report)

```

Run the file                
    terminal --> python customer_prediction.py

Results:

<img src="25 Customer Churn Prediction using classification algorithms/pics/customer-chun-results.png" width="800" />
<br>
<br>

[⬆ Back to content](#content)


## 78. 26: Named Entity Recognition (NER) using spaCy

[⬆ Back to content](#content)

This project demonstrates how to extract and classify named entities such as people, organizations, locations, dates, and monetary values from text using Spacy's pre-trained language model.

It processes sample text and identify relevant entities and visualize them using spaces displacy tool. The extracted entities that we get are organized into pandas data frame and then saved as a CSV file for further analysis.

Now this project highlights key NLP concepts such as text tokenization, entity recognition and data visualization, making it ideal for beginners and data enthusiasts exploring real world applications of natural language processing.

Install required package        
    terminal --> pip install spacy      

Download the pre-trained english model       
    terminal --> python -m spacy download en_core_web_sm

named_entity_recognition.py

```python
# install spacy package
## pip install spacy
# download the english pre-trained model
## python -m spacy download en_core_web_sm

# import libraries

## Spacy library for any R task
import spacy

## Import Displacy, a tool for visualizing NER results.
from spacy import displacy

## Creating and managing our data frames
import pandas as pd

# Load the English model
## This load spacy's small English model which supports NER, POS tagging and more.
nlp = spacy.load("en_core_web_sm")

# Sample text for NER
## This is a multi-line string which contains various named entities, including companies, dates, people, locations and monetary amounts as we can see from the text.
text = """
Amazon announced its quarterly earnings on July 30, 2023.
CEO Andy Jassy said the company is investing $4 billion in AI technology.
Google, based in Mountain View, California, also shared its financial report.
The 2024 Summer Olympics will be held in Paris, France.
"""

# Process the text with spaCy
## This processes the text using the loaded Spacy model, generating a doc object that contains tokens, named entities and linguistic annotations.
doc = nlp(text)

# Function to extract entities - extracts named entity from the processed text which is doc.
def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        ## Contains all the detected named entities from the process text, the documents
        entities.append({
            'Entity': ent.text,             # actual text of the entity
            'Label': ent.label_,            # entity label or example in this case org for organizations.
            'Explanation': spacy.explain(ent.label_)    # provides a human readable explanation of the label
        })
    ## Convert the extracted entities into a pandas DataFrame for easy visualization. So we have everything available which will be returned back.
    return pd.DataFrame(entities)

# Extract named entities into a DataFrame
## entities_df stores the resulting data frame containing entities labels and explanations
entities_df = extract_entities(doc)

# Display extracted entities to the user
## Display the extracted name entities in a tabular format using the DataFrame entities
print("Extracted Named Entities:")
print(entities_df)

# Visualize Named Entities using DisplaCy
## This visualizes the named entity in a web friendly format using display.
## style="ent" specifies that we want to visualize named entities
## jupyter=True ensures a rendering works within a Jupiter notebook.
displacy.render(doc, style="ent", jupyter=True)

# Save entities to a CSV file
## index=False xcludes the data frame index from the CSV file, or else it will have those 0, 1, 2, 3.
entities_df.to_csv("extracted_entities.csv", index=False)
## Confirm that the file was saved successfully.
print("\nEntities saved to 'extracted_entities.csv'")

```

Run the file                
    terminal --> python named_entity_recognition.py

Results:

<img src="26 Named Entity Recognition (NER) using spaCy/pics/named-entities-results.png" width="600" />
<br>
<br>

So it'll give you all that information and also the explanation of each and every one.

Now, they might not always be perfect. For example, AI is not a country, cities or state. So that is wrong.

But it will give you some information about the content that you have.

And also we can check CSV file extracted_entities.csv.

[⬆ Back to content](#content)


## 79. 27: Predict Employee Attrition using XGBoost

[⬆ Back to content](#content)

We will build a predictive model using XGBoost to determine whether an employee is likely to leave a company based on various job related features, such as job satisfaction, years at the company, salary and performance ratings.

You can use whichever you like, but make sure that the columns match or you fix the columns accordingly.

Downoad the dataset from kaggle.com - https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Save the dataset in folder 'data/HR-Employee-Attrition.csv'

Install required packages       
    terminal --> pip install xgboost pandas scikit-learn matplotlib seaborn

predict_employee_attration.py

```python
# Install required packages       
## pip install xgboost pandas scikit-learn matplotlib seaborn

# import libraries

## These are data manipulation and numerical operations XGBoost.
## Data manipulation library allowing you to load and manipulate data in a structured format like a DataFrame
import pandas as pd
## Matrix operations and mathematical functions
import numpy as np

## Xgboost classifier
import xgboost as xgb

## Splits data into training and testing sets
from sklearn.model_selection import train_test_split

## Labelencoder
from sklearn.preprocessing import LabelEncoder

# Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification - displays true false positives and negatives.
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class - detailed performance report of a model.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Used for data visualization
import matplotlib.pyplot as plt

## data visualization - create a heat map from the confusion matrix visualization
import seaborn as sns

# Load the dataset (replace with actual file path)
## If you have a different dataset replace this with your actual file path
df = pd.read_csv('data/HR-Employee-Attrition.csv')

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Preprocess the dataset
# Drop irrelevant columns - removes all the irrelevant columns that don't affect iteration prediction.
df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
## This encodes the categorical features into numerical values using label encoder which is helpful for making sure it doesn't give us error, because the classifier doesn't understand categorical values, it understands numerical values.
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split features and target
## Drop the attrition because that is our feature
X = df.drop('Attrition', axis=1)
## y variable the target variable where 0 is employee stayed and 1 is employee left the company
y = df['Attrition']

# Train-test split
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes (stayed and left) are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the XGBoost classifier
## This initializes the XGBoost classifier and trains it on the training set.
## use_label_encoder=False disables the deprecated label encoder.
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
## fit our model and pass it the data X_train and y_train to train the model.
model.fit(X_train, y_train)

# Make predictions
## This uses the trained model to make predictions on the test set (X_test).
y_pred = model.predict(X_test)

# Evaluate the model
## Calculates the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
## Generates a confusion matrix which we will be displaying using the using matplotlib
conf_matrix = confusion_matrix(y_test, y_pred)
## Provides precision recall F1 score and support in a tabular format
class_report = classification_report(y_test, y_pred)

# This displays all the evaluation metrics for us to see on the command prompt.
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", class_report)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualize the confusion matrix
## So this entire code here will visualize the confusion matrix using Seaborn for better readability than what we have in our text
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

```

Run the file                
    terminal --> python predict_employee_attration.py

Results:

Confusion matrix here, which is displayed using our Matplotlib and Seaborn. We display that data here where it shows stayed vs. left, true positives, true negatives, false positives, and false negative in this confusion matrix here.

<img src="27 Predict Employee Attrition using XGBoost/pics/predict-employee-attraction-results.png" width="600" />
<br>
<br>

<img src="27 Predict Employee Attrition using XGBoost/pics/predict-employee-attraction-results-2.png" width="400" />
<br>
<br>

[⬆ Back to content](#content)


## 80. 28: Disease Prediction (e.g., Heart Disease) using ML algorithms

[⬆ Back to content](#content)

The objective of the project is to build a machine learning model to predict the likelihood of heart disease based on patient health data.

The model will leverage supervised learning algorithms to analyze key health indicators and classify whether a patient is at risk of heart disease.

The steps that we will follow are:
- first we will load and explore the dataset
- second we will preprocess the data.
- third we will split the data into training and testing sets.
- fourth we will train the ML models
- fifth we will evaluate the model performance
- sixth we will make predictions on new data.
- And finally we will visualize the results.

We will use Kaggle dataset - https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Download and set the dataset in folder 'data/heart.csv'

disease_prediction.py

```python
# import libraries

## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## matrix operation and mathematical function
import numpy as np


## Splits data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV

## This scales features to have a mean of zero and standard deviation of one helping improve model stability.
from sklearn.preprocessing import StandardScaler

## Binary classification problems in which in our case is heart disease or no heart disease
from sklearn.linear_model import LogisticRegression

## random forest classifier - ensemble method for classification
from sklearn.ensemble import RandomForestClassifier

## Imports performance metrics to evaluate the model's performance
## Confusion matrix generates a matrix that shows the performance of the classification - displays true false positives and negatives.
## Accuracy score which calculates how often the model's predictions are correct
## Classification report which provides precision, recall, and F1-score for each class - detailed performance report of a model.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Allows us to plot graphs and charts such as the confusion matrix that we will do
import matplotlib.pyplot as plt

## Data visualization - enhances visualization with statistical plots
import seaborn as sns

## We will not see the warnings that are unnecessary for this project.
## If you want to see the warnings you warnings remove this part
import warnings
warnings.filterwarnings('ignore')   


# Step 1: Load Dataset
## If you have a URL or if you're using some other file, make sure that you change that to that particular URL or file
## Data set purpose here is the heart disease data set for predicting heart disease presence
df = pd.read_csv('data/heart.csv')

# Display first few rows
print("Dataset Sample:")
print(df.head())

# Step 2: Data Preprocessing
# Handle missing values (if any)
## df.isnull() identifies missing or null values and .sum() counts the null values in each column
print("\nMissing Values:\n", df.isnull().sum())

# Feature Scaling
## Initializes a scalar
scaler = StandardScaler()
## fit_transform() fits the scaler and scales numerical features
## df.drop('target', axis=1) drops the target column to keep only the feature columns
## We have the last column as target, which we want to drop and just keep the feature columns in this case
scaled_features = scaler.fit_transform(df.drop('target', axis=1))
## X stores scaled features as a DataFrame
X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
## Y stores the target Get variable which is the target column. So Y is our target column.That's what we want to predict. And that's why we keep it in the Y variable.
y = df['target']

# Step 3: Split the Dataset
## X is our feature set, y is our target which is spam
## Test size is out of all the data that we have, we want 20% of the data to be test, whereas 80% of the data to be used as a training data
## This will give us variables - training data, testing data, training targets and testing targets
## stratify=y ensures that both classes (disease and not disease) are proportionally represented in the training and test sets
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train Multiple ML Models
# 1. Logistic Regression
## Initialize logistic regression model
log_model = LogisticRegression()
## Trains the model using the training data
log_model.fit(X_train, y_train)
## Predicts outcomes on the test data
log_preds = log_model.predict(X_test)
## Calculate the accuracy of the logistic model
log_accuracy = accuracy_score(y_test, log_preds)
## Print the logistic regression accuracy
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

# 2. Random Forest Classifier
## Initialize random forest model with 100 decision trees
## random_state=42 - key used to make sure that the split is always the same, no matter how many times how we split it
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
## Trains the model using the training data
rf_model.fit(X_train, y_train)
## This predicts outcome on test data
rf_preds = rf_model.predict(X_test)
## Calculate the accuracy of the random forest model
rf_accuracy = accuracy_score(y_test, rf_preds)
## Print the random forest model accuracy
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Step 5: Evaluate the Best Model of the two
## This is where we do the do the model selection, we choose the model with the higher accuracy
## Find best model
best_model = rf_model if rf_accuracy > log_accuracy else log_model
## find best prediction
best_preds = rf_preds if rf_accuracy > log_accuracy else log_preds

## Print hte best results from both models
print("\nBest Model Metrics:")
print("Accuracy Score:", accuracy_score(y_test, best_preds))
## Classification report again will display precision recall F1 score and support
print("Classification Report:\n", classification_report(y_test, best_preds))
## Confusion matrix displays the confusion matrix values which are true negatives true positives false negatives false positives
print("Confusion Matrix:\n", confusion_matrix(y_test, best_preds))

# Step 6: Visualize Confusion Matrix
## Set the plot size
plt.figure(figsize=(8, 6))
## This will display the confusion matrix as a heat map
## annot=True annotates matrix with the values
## cmap='Blues' - blue colors
sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, cmap='Blues', fmt='d')
## Create the title
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
## This will show us the confusion matrix
plt.show()

# Step 7: Make Predictions on New Data
new_data = pd.DataFrame({
    'age': [45],
    'sex': [1],
    'cp': [2],
    'trestbps': [130],
    'chol': [230],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [0.5],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})

# Scale new data
## This applies the previously fitted scalar to the new data
new_data_scaled = scaler.transform(new_data)
## Predict the outcome for the new data
prediction = best_model.predict(new_data_scaled)
## print out prediction for new data at risk disease if prediction = 0 - heart disease, if prediction = 1 - no heart disease
print("\nPrediction for New Data:", "At Risk of Hearth Disease" if prediction[0] == 1 else "No Hearth Disease")

```

Run the file                
    terminal --> python disease_prediction.py

Results:

<img src="28 Disease Prediction (e.g., Heart Disease) using ML algorithms/pics/hearth-disease-results.png" width="600" />
<br>
<br>


<img src="28 Disease Prediction (e.g., Heart Disease) using ML algorithms/pics/hearth-disease-table-result-1.png" width="800" />
<br>
<br>


<img src="28 Disease Prediction (e.g., Heart Disease) using ML algorithms/pics/hearth-disease-table-result-2.png" width="300" />
<br>
<br>

Logistic regression accuracy is 0.81 whereas random forest accuracy is 100% which is pretty good.       

<img src="28 Disease Prediction (e.g., Heart Disease) using ML algorithms/pics/hearth-disease-table-result-3.png" width="400" />
<br>
<br>

This is a classification report based on that which is perfect 100 over 100. So that's good.

Next we have the confusion matrix which we already saw in a graph format which was nice. And then finally we have the prediction made here based on the data that we provided.

[⬆ Back to content](#content)


## 81. 29: Movie Rating Prediction using Collaborative Filtering

[⬆ Back to content](#content)

The project overview here is to build a movie rating prediction system using collaborative filter where the model predicts how a user would rate a movie based on historical user movie interactions.

The goal here is to predict movie ratings for users and the approach that we'll be using is collaborative filtering with techniques like matrix factorization example singular value decomposition also known as SVD.

Download small dataset from https://grouplens.org/datasets/movielens/ - ml-latest-small.zip (size: 1 MB). We will use only one of the files inside the dataset - ratings.csv

Save it in folder 'data/ratings.csv'

In the dataset we have four different rows. One is user ID which is unique identifier for users. We have movie ID which is unique identifier for the movies. We have rating which is rating given by a user which is one from 1 to 5 depending on that. And then we have the timestamp which is time of the rating. Not essential for this project, but that's available. The timestamp doesn't matter as we can imagine. 

Install packages        
    terminal --> pip install surprise "numpy<2"

movie_rating_prediction.py

```python
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

```

Run the file                
    terminal --> python movie_rating_prediction.py

Results:

<img src="29 Movie Rating Prediction using Collaborative Filtering/pics/movie-rating-result-1.png" width="800" />
<br>
<br>

<img src="29 Movie Rating Prediction using Collaborative Filtering/pics/movie-rating-result-2.png" width="400" />
<br>
<br>


[⬆ Back to content](#content)


## 82. 30: Automatic Essay Grading using BERT

[⬆ Back to content](#content)

The project overview is to build an automatic essay grading system using Bert which is bidirectional encoder representations from transformers to evaluate and assign scores to essays based on their content, coherence and relevance. The prerequisites here are pandas, scikit learn, torch, matplotlib and Seaborn.

We'll use the Automated Student Assessment Prize ASAP dataset available on Kaggle. The key features in the dataset is essay and score. - https://www.kaggle.com/datasets/lburleigh/asap-2-0

Download and store the data in folder 'data/ASAP2_train_sourcetexts.csv'

Install packages        
    terminal --> pip install pandas scikit-learn torch matplotlib seaborn

essay_grading_system.py

```python
# import libraries

## Data manipulation library allowing you to load and manipulate data in a structured format like a data frame
import pandas as pd

## matrix operation and mathematical function
import numpy as np

## Splits the data set into training and testing sets
from sklearn.model_selection import train_test_split

## This standardizes the numerical features
from sklearn.preprocessing import StandardScaler

## mean_squared_error and R2_score are metrics used to evaluate the performance of the model
## mean_squared_error measures the mean squared error of prediction
## r2_score measures the proportion of variance explained by the model
from sklearn.metrics import mean_squared_error, r2_score

## core plotting library for visualizations
import matplotlib.pyplot as plt

## Advanced statistical plot which is an advanced statistical plots library
import seaborn as sns

## PyTorch library for building neural networks
import torch

## Dataset is an abstract class for data
## DataLoader efficiently load data into batches for training
from torch.utils.data import Dataset, DataLoader

## BertTokenizer - Tokenizer for text preprocessing using Bert
## BertModel - pre-trained Bert model for embeddings
## AdamW - optimizer for training the neural network
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

# Step 1: Load and Explore the Dataset
df = pd.read_csv('data/ASAP2_train_sourcetexts.csv')
print("Dataset Sample:")
print(df.head())

# Select relevant columns
# df = df[['Essay', 'Overall']]
df = df[['full_text', 'score']]

# Step 2: Preprocess the Data
## This loads the pre-trained Bert tokenizer for lowercase English text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset Class
class EssayDataset(Dataset):
    ## different variables that we will initialize
    def __init__(self, essays, scores, tokenizer, max_len=512):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    ## returns the number of essays in the data set
    def __len__(self):
        return len(self.essays)

    ## retrieves a specific essay and score tokenizes the text and converts data into tensors
    def __getitem__(self, index):
        essay = str(self.essays[index])
        score = self.scores[index]
        
        encoding = self.tokenizer(
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        ## returns input IDs which are numerical IDs of tokens, attention mask which is the identifiers real token versus padding, and score which converts essay score into a PyTorch tensor
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }

# Split Dataset
## test_size=0.2 split splits data into 80% training and 20% validation based on my specification
## ## random_state=42 ensures reproducibility - Every time you use this number, it will give the exact same training set and test set
train_texts, val_texts, train_scores, val_scores = train_test_split(
    df['full_text'], df['score'], test_size=0.2, random_state=42
)

# This creates instances of the SAR dataset class for training and validation
## set train dataset with the different essay parameters
train_dataset = EssayDataset(train_texts.tolist(), train_scores.tolist(), tokenizer)
## set validation set with the same parameters
val_dataset = EssayDataset(val_texts.tolist(), val_scores.tolist(), tokenizer)

# The data loader efficiently batches data during training and validation.
## batch_size=8 processes eight samples per iteration. We can increase or decrease depending on how much you want to do.
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Step 3: Build the Model
class EssayGradingModel(torch.nn.Module):
    def __init__(self):
        super(EssayGradingModel, self).__init__()
        ## loading the pre-trained Bert model which extracts the embeddings
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        ## This is a linear layer which reduces Bert's output to a single score
        self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)

    ## This forward process is the forward for processing input tensors and returns predicted scores
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        score = self.regressor(cls_output)
        return score.squeeze()

# Initialize Model, Optimizer, and Loss Function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EssayGradingModel().to(device)
## This is optimizer for training
optimizer = AdamW(model.parameters(), lr=2e-5)
## MSE loss measures mean squared error loss
loss_fn = torch.nn.MSELoss()

# Step 4: Train the Model
def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    ## for each and every batch that we have created
    for batch in data_loader:
        ## this will zero out the gradients from before
        optimizer.zero_grad()
        ## input IDs to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    ## performs the forward pass, Calculates the loss and updates the weights
    return total_loss / len(data_loader)

# Step 5: Evaluate the Model
## Evaluates model performance using MSE and R2
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    predictions = []
    true_scores = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, scores)
            total_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            true_scores.extend(scores.cpu().numpy())

    mse = mean_squared_error(true_scores, predictions)
    r2 = r2_score(true_scores, predictions)
    ## three parameters will pass back
    return total_loss / len(data_loader), mse, r2

# Training Loop
epochs = 3              # can take some time on start
for epoch in range(epochs):
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    val_loss, mse, r2 = evaluate(model, val_loader, loss_fn, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}')

# Step 6: Test the Model
test_text = "The essay provided insightful analysis of the topic with well-structured argument."
encoding = tokenizer(test_text, add_special_tokens=True, max_length=512, padding='max_length', return_tensors='pt')
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

## This will evaluate and test the model and will give us the outputs accordingly
model.eval()

with torch.no_grad():
    predicted_score = model(input_ids, attention_mask).item()

## This will give us a predicted score based on the custom essay that we have above
print(f"Predicted Score: {predicted_score:.2f}")

# Step 7: Visualize Training Performance
## Plot training and validation loss over the epochs
losses = {'Epoch': [1, 2, 3], 'Train Loss': [0.3, 0.2, 0.1], 'Val Loss': [0.35, 0.25, 0.15]}
loss_df = pd.DataFrame(losses)
## display a line plot, which will look like more of a slant line for both the training and validation
sns.lineplot(data=loss_df, x='Epoch', y='Train Loss', label='Train Loss')
sns.lineplot(data=loss_df, x='Epoch', y='Val Loss', label='Val Loss')
plt.title('Training and Validation Loss')
plt.show()

```

It may take from 30 minutes to 3-4 hours because Bert is slow model.

Run the file                
    terminal --> python essay_grading_system.py

Results:

[⬆ Back to content](#content)

