# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.config import DIMENSIONS


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
    def __init__(self, input_dim, num_classes, heads=8, dim_feedforward=512):
        super(AttentionBasedModel, self).__init__()
        self.self_attention = SelfAttention(input_dim, heads)
        self.fc1 = nn.Linear(input_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output = self.self_attention(x)
        attn_output = self.norm(attn_output + x)
        pooled_output = torch.mean(attn_output, dim=1)
        x = F.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
