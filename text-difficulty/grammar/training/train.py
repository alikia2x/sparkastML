import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# Load the model classes
from model import AttentionBasedModel

# Load the data
df = pd.read_csv('data.csv')

# Step 1: Extract sentences and corresponding levels
sentences = df['Sentence'].values
levels = df['Level'].values

# Step 2: Tokenize and POS tag each sentence
pos_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences]

# Step 3: Build POS tag vocabulary
# Extract unique POS tags from the dataset
pos_vocab = set()
for tagged_sentence in pos_tags:
    for _, tag in tagged_sentence:
        pos_vocab.add(tag)

# Create a mapping from POS tag to index
pos2idx = {pos: idx for idx, pos in enumerate(pos_vocab)}
pos_vocab_size = len(pos2idx)

# Step 4: Encode sentences into POS tag indices
def encode_pos_tags(tagged_sentence):
    return [pos2idx[tag] for _, tag in tagged_sentence]

encoded_sentences = [encode_pos_tags(tagged_sentence) for tagged_sentence in pos_tags]

# Step 5: Encode levels (classes) into integers
le = LabelEncoder()
encoded_levels = le.fit_transform(levels)
num_classes = len(le.classes_)

# Save class encoding mapping
class_mapping = dict(zip(le.transform(le.classes_), le.classes_))
torch.save(class_mapping, 'class_mapping.pt')

# Save POS tag encoding mapping
torch.save(pos2idx, 'pos2idx.pt')

# Step 6: Pad sentences to a fixed length
max_length = 64

def pad_sequence(seq, max_len):
    return seq + [-1] * (max_len - len(seq))

padded_sentences = [pad_sequence(seq, max_length) for seq in encoded_sentences]

# Step 7: Create a PyTorch Dataset and DataLoader
class POSDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = torch.tensor(self.sentences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sentence, label

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sentences, encoded_levels, test_size=0.2)

train_dataset = POSDataset(X_train, y_train)
val_dataset = POSDataset(X_val, y_val)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 8: Initialize the model, loss function, and optimizer
embedding_dim = 32
heads = 8
num_attention_layers = 6
dim_feedforward = 256
learning_rate = 0.003

model = AttentionBasedModel(pos_vocab_size, embedding_dim, num_classes, heads, num_attention_layers, dim_feedforward)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 9: Training loop
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

step = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for sentences, labels in train_loader:
        sentences, labels = sentences.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(sentences)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        
        running_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sentences, labels in val_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print training and validation stats
    print(f'Epoch [{epoch+1}/{num_epochs}], Step {step}, Loss: {running_loss/len(train_loader):.4f}, '
          f'Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    torch.save(model.state_dict(), f'checkpoints/step_{step}.pt')

# Step 10: Save the trained model
torch.save(model.state_dict(), 'model.pt')
