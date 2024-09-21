# data_utils.py

import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def create_class_mappings(data):
    class_to_idx = {cls: idx for idx, cls in enumerate(data.keys())}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def preprocess_data(data, embedding_map, tokenizer, class_to_idx, max_length=64):
    dataset = []
    for label, sentences in data.items():
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            embeddings = [
                embedding_map[token_id] for token_id in token_ids[:max_length]
            ]
            embeddings = torch.tensor(embeddings)
            dataset.append((class_to_idx[label], embeddings))
    return dataset


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        labels, embeddings = zip(*batch)
        labels = torch.tensor(labels)
        embeddings = pad_sequence(embeddings, batch_first=True)
        return labels, embeddings


class NegativeSampleDataset(Dataset):
    def __init__(self, negative_samples):
        self.samples = negative_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        embeddings = pad_sequence(batch, batch_first=True)
        return embeddings
