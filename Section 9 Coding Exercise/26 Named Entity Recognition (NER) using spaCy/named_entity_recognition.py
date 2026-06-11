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
