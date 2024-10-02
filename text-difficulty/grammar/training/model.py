# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding matrix of shape (max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension, so the shape becomes (1, max_len, embedding_dim)
        pe = pe.unsqueeze(0)
        
        # Register the positional encoding as a buffer so it won't be updated by the optimizer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape (batch_size, seq_length, embedding_dim)
        seq_length = x.size(1)
        # Add positional encoding to input
        x = x + self.pe[:, :seq_length]
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.scale = (input_dim // heads) ** -0.5
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.shape
        qkv = self.qkv(x).view(
            batch_size, seq_length, self.heads, 3, embedding_dim // self.heads
        )
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attention_output = torch.matmul(attn_weights, v)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, embedding_dim)
        
        return self.fc(attention_output)

class AttentionBasedModel(nn.Module):
    def __init__(self, pos_vocab_size, embedding_dim=128, num_classes=6, heads=8, num_attention_layers=3, dim_feedforward=512, max_len=128):
        super(AttentionBasedModel, self).__init__()
        self.embedding = nn.Embedding(pos_vocab_size, embedding_dim)  # Embedding for POS tags
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)  # Positional Encoding
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(embedding_dim, heads) for _ in range(num_attention_layers)
        ])
        self.fc1 = nn.Linear(embedding_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Input x is a matrix of one-hot encoded POS tags, shape: (batch_size, seq_length, pos_vocab_size)
        x = self.embedding(x)  # Convert POS tags to embeddings, shape: (batch_size, seq_length, embedding_dim)
        
        # Add positional encoding to embeddings
        x = self.positional_encoding(x)
        
        for attn_layer in self.self_attention_layers:
            attn_output = attn_layer(x)
            x = self.norm(attn_output + x)

        # Pool the output by taking the mean of the sequence (reduce along sequence length)
        pooled_output = torch.mean(x, dim=1)

        # Fully connected layers for classification
        x = F.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits for the 6 classes
        
        return x


# Example Usage
# # Hyperparameters
# pos_vocab_size = 50  # Size of the POS tag vocabulary
# max_context_length = 128  # Maximum context length
# embedding_dim = 128  # Embedding size
# num_classes = 6  # Output classes
# batch_size = 32  # Example batch size

# # Model initialization
# model = AttentionBasedModel(pos_vocab_size, embedding_dim, num_classes)

# # Example input: batch of one-hot encoded POS tags (variable length sequences)
# input_data = torch.randint(0, pos_vocab_size, (batch_size, max_context_length))  # Random input for testing

# # Forward pass
# output = model(input_data)  # Output shape will be (batch_size, num_classes)

# print(output.shape)  # Should print torch.Size([batch_size, num_classes])
