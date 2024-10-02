from training.model import AttentionBasedModel
import torch
import torch.nn.functional as F
import nltk


sentence = '''Smartphones have worked their way deep into our lives and have become indispensable for work and socialising.'''

pos2idx = torch.load('pos2idx.pt')
class_mapping = torch.load('class_mapping.pt')

def pad_sequence(seq, max_len):
    return seq + [-1] * (max_len - len(seq))

def encode_pos_tags(tagged_sentence):
    return [pos2idx[tag] if tag in pos2idx else -1 for _, tag in tagged_sentence]


max_length = 64

tagged_sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
encoded_sentences = encode_pos_tags(tagged_sentence)
padded_sentence = torch.tensor(pad_sequence(encoded_sentences, max_length))

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = AttentionBasedModel(40, 32, 6, 8, 6, 256)
model.load_state_dict(torch.load("./model.pt", weights_only=True))
model.to(device)
model.eval()  # 确保模型处于评估模式


w_list=[1.35, 1.63, 2.75, 3.64, 5.38, 6.32]
cefr_dict = [None, "A1", "A2", "B1", "B2", "C1", "C2"]
with torch.no_grad():
    sentence = padded_sentence.to(device)
    sentences = torch.unsqueeze(sentence, 0)
    
    outputs = model(sentences)
    print(torch.max(outputs, 1))
    print(outputs[0])
    print(torch.softmax(outputs[0],0))
    s=0
    for i in range(6):
        s+=torch.softmax(outputs[0],0)[i] * w_list[i]
    s=s.cpu().numpy()
    # the s is the final output.