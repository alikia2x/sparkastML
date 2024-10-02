import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from training.model import AttentionBasedModel

# Ensure required NLTK resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load pre-saved mappings
pos2idx = torch.load('pos2idx.pt')
class_mapping = torch.load('class_mapping.pt')


# Load the pre-trained model and state
model = AttentionBasedModel(40, 32, 6, 8, 6, 256)
model.load_state_dict(torch.load("./model.pt", weights_only=True))

# Define helper functions
def pad_sequence(seq, max_len):
    return seq + [-1] * (max_len - len(seq))

def encode_pos_tags(tagged_sentence):
    return [pos2idx[tag] if tag in pos2idx else -1 for _, tag in tagged_sentence]

# Split sentence into smaller chunks based on punctuation and length constraints
def split_long_sentence(sentence, max_len=128):
    tokens = word_tokenize(sentence)
    
    if len(tokens) <= max_len:
        return [sentence]
    
    # Attempt to split based on punctuation marks
    punctuation_marks = [',', ';', ':', '!', '?', '.', '-']
    split_chunks = []
    current_chunk = []
    
    for token in tokens:
        current_chunk.append(token)
        if token in punctuation_marks and len(current_chunk) >= max_len // 2:
            split_chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        split_chunks.append(' '.join(current_chunk))
    
    # If chunks are still too long, truncate them
    final_chunks = []
    for chunk in split_chunks:
        chunk_tokens = word_tokenize(chunk)
        if len(chunk_tokens) > max_len:
            final_chunks.extend([' '.join(chunk_tokens[i:i + max_len]) for i in range(0, len(chunk_tokens), max_len)])
        else:
            final_chunks.append(chunk)
    
    return final_chunks

# Main function to process and score a chunk
def score_sentence(sentence, model, max_length=128):
    # Tokenize and POS-tag the sentence
    tagged_sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
    
    # Encode the POS tags and pad the sequence
    encoded_sentences = encode_pos_tags(tagged_sentence)
    padded_sentence = torch.tensor(pad_sequence(encoded_sentences, max_length))
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Prepare the model
    model.to(device)
    model.eval()  # Ensure the model is in evaluation mode

    # Define weights and CEFR levels
    w_list = [1.04, 1.64, 2.35, 3.44, 4.92, 6.13]
    
    # Inference without gradient calculation
    with torch.no_grad():
        sentence_tensor = padded_sentence.to(device)
        sentence_tensor = torch.unsqueeze(sentence_tensor, 0)  # Add batch dimension
        
        # Forward pass through the model
        outputs = model(sentence_tensor)
        
        # Softmax and weighted scoring
        probabilities = torch.softmax(outputs[0], dim=0)
        score = sum(probabilities[i] * w_list[i] for i in range(6)).cpu().numpy()
        
    return score

# Function to process a long article and return score list for each chunk
def score_article(article, max_length=128, chunk_max_len=128):
    sentences = sent_tokenize(article)  # Split the article into sentences
    score_list = []
    
    for sentence in sentences:
        chunks = split_long_sentence(sentence, max_len=chunk_max_len)
        for chunk in chunks:
            score = score_sentence(chunk, model, max_length=max_length)
            score_list.append(float(score))
    
    return score_list
