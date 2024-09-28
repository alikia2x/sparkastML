from training.model import AttentionBasedModel
from training.config import model_name
from training.config import DIMENSIONS
from training.data_utils import get_sentences
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

def energy_score(logits):
    # Energy score is minus logsumexp
    return -torch.logsumexp(logits, dim=1)


def get_energy(
    model,
    sentence,
    embedding_map,
    tokenizer,
    max_length=64,
):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
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
        # Calculate energy score
        energy = energy_score(logits)

    return energy


with open("data.json", "r") as f:
    positive_data = json.load(f)
class_to_idx = {cls: idx for idx, cls in enumerate(positive_data.keys())}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
num_classes = len(class_to_idx)

with open("noise.json", "r") as f:
    negative_data = json.load(f)

input_dim = DIMENSIONS
model = AttentionBasedModel(input_dim, num_classes)
model.load_state_dict(torch.load("./model.pt"))
embedding_map = torch.load("token_id_to_reduced_embedding.pt")
tokenizer = AutoTokenizer.from_pretrained(model_name)


all_preds = []
all_labels = []
ENERGY_THRESHOLD = 2
for item in tqdm(get_sentences(positive_data)):
    result = get_energy(model, item, embedding_map, tokenizer) < ENERGY_THRESHOLD
    all_preds.append(result)
    all_labels.append(1)

for item in tqdm(negative_data):
    result = get_energy(model, item, embedding_map, tokenizer) < ENERGY_THRESHOLD
    all_preds.append(result)
    all_labels.append(0)
    
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
accuracy = accuracy_score(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')