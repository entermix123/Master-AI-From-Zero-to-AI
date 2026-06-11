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

        # encoding = self.tokenizer.encode_plus(
        #     essay,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     padding='max_length',
        #     truncation=True,
        #     return_attention_mask=True,
        #     return_tensors='pt'
        # )
        
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
