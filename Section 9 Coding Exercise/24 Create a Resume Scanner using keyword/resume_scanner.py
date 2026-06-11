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
