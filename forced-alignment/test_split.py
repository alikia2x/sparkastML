import torch
import torchaudio
from typing import List
from pypinyin import lazy_pinyin
from pypinyin_dict.phrase_pinyin_data import cc_cedict
from torchaudio.transforms import Resample


def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

# Compute average score weighted by the span length
def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchaudio.pipelines import MMS_FA as bundle

model = bundle.get_model()
model.to(device)

tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

cc_cedict.load()

add_spaces = lambda s: ' '.join(s)

with open("./segments/line-1.txt", "r") as f:
    text = f.read()

text_raw = add_spaces(text)
text_list = list(text)
text_pinyin = lazy_pinyin(text)
text_normalized = " ".join(text_pinyin)

waveform, sample_rate = torchaudio.load("./segments/line-1.wav")

waveform = waveform[0:1]
resampler = Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

transcript = text_normalized.split()
emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)


print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)

ratio = waveform.size(1) / num_frames

for i in range(len(token_spans)):
    spans = token_spans[i]
    x0 = round(int(ratio * spans[0].start) / 16000, 3)
    x1 = round(int(ratio * spans[-1].end) / 16000, 3)
    print(f"{text[i]}: {x0}-{x1}")