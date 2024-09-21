from training.model import AttentionBasedModel
from training.config import model_name
import json
import torch
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from training.config import DIMENSIONS
from training.model import AttentionBasedModel


def energy_score(logits):
    # Energy score is minus logsumexp
    return -torch.logsumexp(logits, dim=1)


def predict_with_energy(
    model,
    sentence,
    embedding_map,
    tokenizer,
    idx_to_class,
    energy_threshold,
    max_length=64,
):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)
    embeddings = [embedding_map[token_id] for token_id in token_ids[:max_length]]
    embeddings = torch.tensor(embeddings).unsqueeze(0)  # Add batch dimension
    current_shape = embeddings.shape
    
    if current_shape[1] < 2:
        pad_size = 2 - current_shape[1]
        embeddings = F.pad(
            embeddings, (0, 0, 0, pad_size, 0, 0), mode="constant", value=0
        )

    with torch.no_grad():
        logits = model(embeddings)
        print(logits)
        probabilities = F.softmax(logits, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

        # Calculate energy score
        energy = energy_score(logits)

    # If energy > threshold, consider the input as unknown class
    if energy.item() > energy_threshold:
        return ["Unknown", max_prob.item(), energy.item()]
    else:
        return [idx_to_class[predicted.item()], max_prob.item(), energy.item()]


with open("data.json", "r") as f:
    data = json.load(f)
class_to_idx = {cls: idx for idx, cls in enumerate(data.keys())}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
num_classes = len(class_to_idx)

input_dim = DIMENSIONS
model = AttentionBasedModel(input_dim, num_classes)
model.load_state_dict(torch.load("./model.pt"))
embedding_map = torch.load("token_id_to_reduced_embedding.pt")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example usage:
ENERGY_THRESHOLD = 0
sentence = "天气"
energy_threshold = ENERGY_THRESHOLD
predicted = predict_with_energy(
    model, sentence, embedding_map, tokenizer, idx_to_class, energy_threshold
)
print(f"Predicted: {predicted}")
