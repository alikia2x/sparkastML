# train.py

from sklearn.model_selection import train_test_split
import torch
import json
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from training.data_utils import (
    load_data,
    create_class_mappings,
    preprocess_data,
    TextDataset,
    NegativeSampleDataset,
)
from training.model import AttentionBasedModel
import torch.nn as nn


def energy_score(logits, temperature=1.0):
    return -torch.logsumexp(logits / temperature, dim=1)


def generate_noise(batch_size, seq_length, input_dim, device):
    return torch.randn(batch_size, seq_length, input_dim).to(device)


def train_energy_model(
    model,
    train_loader,
    negative_loader,
    criterion,
    optimizer,
    num_epochs=10,
    margin=1.0,
    temperature=0.4,
):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    negative_iter = iter(negative_loader)

    for epoch in range(num_epochs):
        total_loss = 0
        for _, (labels, embeddings) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            class_loss = criterion(outputs, labels)
            known_energy = energy_score(outputs, temperature)
            positive_margin = 0.0
            energy_loss_known = F.relu(known_energy - positive_margin).mean()

            noise_embeddings = torch.randn_like(embeddings).to(device)
            noise_outputs = model(noise_embeddings)
            noise_energy = energy_score(noise_outputs, temperature)
            energy_loss_noise = F.relu(margin - noise_energy).mean()

            try:
                negative_samples = next(negative_iter)
            except StopIteration:
                negative_iter = iter(negative_loader)
                negative_samples = next(negative_iter)
            negative_samples = negative_samples.to(device)
            negative_outputs = model(negative_samples)
            negative_energy = energy_score(negative_outputs, temperature)
            energy_loss_negative = F.relu(margin - negative_energy).mean()

            total_energy_loss = (
                energy_loss_known + energy_loss_noise + energy_loss_negative
            )
            total_loss_batch = class_loss + total_energy_loss

            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def main():
    from config import model_name, DIMENSIONS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = load_data("data.json")
    class_to_idx, idx_to_class = create_class_mappings(data)
    embedding_map = torch.load("token_id_to_reduced_embedding.pt")
    dataset = preprocess_data(data, embedding_map, tokenizer, class_to_idx)
    train_data, _ = train_test_split(dataset, test_size=0.2)

    train_dataset = TextDataset(train_data)

    train_loader = DataLoader(
        train_dataset, batch_size=24, shuffle=True, collate_fn=train_dataset.collate_fn
    )

    with open("noise.json", "r") as f:
        negative_samples_list = json.load(f)

    negative_embedding_list = []
    for sentence in negative_samples_list:
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        embeddings = [embedding_map[token_id] for token_id in token_ids[:64]]
        embeddings = torch.tensor(embeddings)
        negative_embedding_list.append(embeddings)

    negative_dataset = NegativeSampleDataset(negative_embedding_list)
    negative_loader = DataLoader(
        negative_dataset,
        batch_size=24,
        shuffle=True,
        collate_fn=negative_dataset.collate_fn,
    )

    input_dim = DIMENSIONS
    num_classes = len(class_to_idx)
    model = AttentionBasedModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=7e-4)

    train_energy_model(
        model, train_loader, negative_loader, criterion, optimizer, num_epochs=120
    )

    torch.save(model.state_dict(), "model.pt")

    dummy_input = torch.randn(1, 64, DIMENSIONS)
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_length"},
            "output": {0: "batch_size"},
        },
        opset_version=11,
    )
    meta = {
        "idx_to_class": idx_to_class,
        "threshold": 0
    }
    with open('NLU_meta.json', 'w') as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
