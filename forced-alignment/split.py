import torch
import torchaudio
from typing import List
from pypinyin import lazy_pinyin
from pypinyin_dict.phrase_pinyin_data import cc_cedict
from torchaudio.transforms import Resample
from tqdm import tqdm 


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

add_spaces = lambda s: ' '.join(s)

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
    将浮点数的秒钟转换为TTML的时间戳格式（HH:MM:SS.sss）。

    :param seconds: 浮点数的秒钟
    :return: TTML时间戳格式字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)

    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def timestamp_inverse(ttml_timestamp):
    """
    将TTML的时间戳格式字符串（HH:MM:SS.sss）转换为浮点数的秒钟。

    :param ttml_timestamp: TTML时间戳格式字符串
    :return: 浮点数的秒钟
    """
    parts = ttml_timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_and_milliseconds = parts[2].split('.')
    seconds = int(seconds_and_milliseconds[0])
    milliseconds = int(seconds_and_milliseconds[1])

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    return total_seconds

import xml.etree.ElementTree as ET

import os
import re

def extract_numbers_from_files(directory):
    """
    读取给定目录，提取文件名中的数字部分，并返回一个包含这些数字的列表。

    :param directory: 目录路径
    :return: 包含数字的列表
    """
    numbers = []
    pattern = re.compile(r'line-(\d+)\.wav')

    try:
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                numbers.append(number)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return None

    return numbers
    
class TTMLGenerator:
    def __init__(self, duration, xmlns="http://www.w3.org/ns/ttml", xmlns_ttm="http://www.w3.org/ns/ttml#metadata", xmlns_amll="http://www.example.com/ns/amll", xmlns_itunes="http://music.apple.com/lyric-ttml-internal"):
        self.tt = ET.Element("tt", attrib={
            "xmlns": xmlns,
            "xmlns:ttm": xmlns_ttm,
            "xmlns:amll": xmlns_amll,
            "xmlns:itunes": xmlns_itunes
        })
        self.head = ET.SubElement(self.tt, "head")
        self.metadata = ET.SubElement(self.head, "metadata")
        self.body = ET.SubElement(self.tt, "body", attrib={"dur": duration})
        self.div = ET.SubElement(self.body, "div")

    def add_lyrics(self, begin, end, agent, itunes_key, words):
        p = ET.SubElement(self.div, "p", attrib={
            "begin": begin,
            "end": end,
            "ttm:agent": agent,
            "itunes:key": itunes_key
        })
        for word, start, stop in words:
            span = ET.SubElement(p, "span", attrib={"begin": start, "end": stop})
            span.text = word

    def save(self, filename):
        tree = ET.ElementTree(self.tt)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

duration = get_audio_duration("./data/谷雨.mp3")

# 示例使用
ttml_generator = TTMLGenerator(duration=timestamp(duration))


def process_line(line_idx, start_time, total_lines):
    with open(f"./segments/line-{line_idx}.txt", "r") as f:
        text = f.read()
        
    waveform, sample_rate = torchaudio.load(f"./segments/line-{line_idx}.wav")

    waveform = waveform[0:1]
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    text_pinyin = lazy_pinyin(text)
    text_normalized = " ".join(text_pinyin)
    
    transcript = text_normalized.split()
    emission, token_spans = compute_alignments(waveform, transcript)
    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames

    words = []
    for i in range(len(token_spans)):
        spans = token_spans[i]
        x0 = start_time + int(ratio * spans[0].start) / 16000
        x1 = start_time + int(ratio * spans[-1].end) / 16000
        words.append({
            "word": text[i],
            "start": x0,
            "end": x1
        })
    idx=0
    for item in words:
        if idx == len(words) - 1:
            break
        item["end"] = words[idx + 1]["start"]
        idx+=1
    result = []
    for word in words:
        result.append((word["word"], timestamp(word["start"]), timestamp(word["end"])))
    return result


lines_to_process = sorted(extract_numbers_from_files("segments"))

def parse_lrc(lrc_file, audio_len):
    """解析LRC文件，返回一个包含时间戳和歌词的列表"""
    with open(lrc_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lrc_data = []
    for line in lines:
        # 使用正则表达式匹配时间戳和歌词
        match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            lyric = match.group(3).strip()
            lyric = lyric.replace(" ", "")
            timestamp = minutes * 60 + seconds
            lrc_data.append((lyric, timestamp))
    
    for i, (lyric, start_time) in enumerate(lrc_data):
        # Skip empty line
        if lyric.strip() == "":
            continue
        if i < len(lrc_data) - 1:
            end_time = lrc_data[i + 1][1]
        else:
            end_time = audio_len
        lrc_data[i] = (lyric, start_time, end_time)
    
    # Filter empty lines again
    lrc_data = [line for line in lrc_data if line[0].strip() != ""]

    return lrc_data

lrc_data = parse_lrc("./data/谷雨.lrc", duration)

i=0
for line_num in tqdm(lines_to_process):
    start_time = lrc_data[i][1]
    end_time = lrc_data[i][2]
    result = process_line(line_num, start_time, len(lines_to_process))
    ttml_generator.add_lyrics(
        begin=timestamp(start_time), end=timestamp(end_time), agent="v1", itunes_key=f"L{i+1}",
        words=result
    )
    i+=1

# 保存文件
ttml_generator.save("output.ttml")