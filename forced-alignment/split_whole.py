import torch
import torchaudio
from typing import List
from pypinyin import lazy_pinyin
from pypinyin_dict.phrase_pinyin_data import cc_cedict
from torchaudio.transforms import Resample
import xml.etree.ElementTree as ET
import argparse

from pydub import AudioSegment

def get_audio_duration(file_path):
    """
    读取音频文件并获取其时长（秒数）。

    :param file_path: 音频文件的路径
    :return: 音频文件的时长（秒数）
    """
    try:
        audio = AudioSegment.from_file(file_path)
        duration_in_seconds = len(audio) / 1000.0
        return duration_in_seconds
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None

def timestamp(seconds):
    """
    将浮点数的秒钟转换为SRT的时间戳格式（HH:MM:SS,sss）。

    :param seconds: 浮点数的秒钟
    :return: SRT时间戳格式字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchaudio.pipelines import MMS_FA as bundle

model = bundle.get_model()
model.to(device)

tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

cc_cedict.load()

with open("./data/扬旗鸣鼓.txt", "r") as f:
    text = f.read()

text_pinyin = lazy_pinyin(text)
text_normalized = " ".join(text_pinyin)

waveform, sample_rate = torchaudio.load("./data/扬旗鸣鼓.wav")

waveform = waveform[0:1]
resampler = Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

transcript = text_normalized.split()
emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)

ratio = waveform.size(1) / num_frames

duration = get_audio_duration("./data/扬旗鸣鼓.wav")

for i in range(len(token_spans)):
    spans = token_spans[i]
    x0 = round(int(ratio * spans[0].start) / 16000, 3)
    x1 = round(int(ratio * spans[-1].end) / 16000, 3)
    with open("1.srt", "a") as f:
        f.write(f"{i+1}\n")
        f.write(f"{timestamp(x0)} --> {timestamp(x1)}\n")
        f.write(f"{transcript[i]}\n\n")
        f.flush()