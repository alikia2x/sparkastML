import re

def srt_to_lrc(srt_text):
    # 使用正则表达式匹配时间戳和内容
    # who the fuck knows this
    srt_text+='\n\n'
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)\n\n', re.DOTALL)
    matches = pattern.findall(srt_text)
    lrc_lines = []
    
    for start_time, end_time, content in matches:
        # 提取开始时间的高亮字符
        highlight_char = re.search(r'<font color="#00ff00">(.+?)</font>', content)
        if highlight_char:
            highlight_char = highlight_char.group(1)
        else:
            continue
        
        # 将时间戳转换为LRC格式
        f,start_minutes, start_seconds, start_milliseconds = map(int, start_time.replace(',', ':').split(':'))
        f,end_minutes, end_seconds, end_milliseconds = map(int, end_time.replace(',', ':').split(':'))
        
        start_time_lrc = f"{start_minutes:02d}:{start_seconds:02d}.{start_milliseconds:02d}"
        end_time_lrc = f"{end_minutes:02d}:{end_seconds:02d}.{end_milliseconds:02d}"
        
        # 构建LRC行
        lrc_line = f"{highlight_char}|{start_time_lrc},{end_time_lrc}"
        lrc_lines.append(lrc_line)
        
        # 如果内容中有换行符，将其替换为空格
        lrc_line = lrc_line.replace('\n', ' ')
    
    return '\n'.join(lrc_lines)

with open('./data/谷雨.srt', 'r') as f:
    srt_text = f.read()

whole = srt_text.splitlines()[2].replace('<font color="#00ff00">','').replace('</font>','')
whole = whole.replace(' ','\n')
lines = whole.splitlines()

lyric_text = ""
raw_text = srt_to_lrc(srt_text)
raw_lines = raw_text.splitlines()
for line in raw_lines:
    lyric_text += line.split('|')[0]

raw_idx=0
lines_start_chr_idx=[]
for line in lines:
    start = line[0]
    end = line[-1]
    while raw_idx < len(raw_lines) and not line.startswith(raw_lines[raw_idx].split("|")[0]):
        raw_idx += 1
    lines_start_chr_idx.append(raw_idx)
lines_start_chr_idx.append(len(raw_lines)-1)

raw_idx=0
lines_end_chr_idx=[]
for line in lines:
    start = line[0]
    end = line[-1]
    while raw_idx < len(raw_lines) and not line.endswith(raw_lines[raw_idx].split("|")[0]):
        raw_idx += 1
    lines_end_chr_idx.append(raw_idx)
lines_end_chr_idx.append(len(raw_lines)-1)

lrc_text = ""
for i in range(len(lines_start_chr_idx)-1):
    start = lines_start_chr_idx[i]
    end = lines_end_chr_idx[i]
    time_start = raw_lines[start].split("|")[1].split(',')[0]
    time_end = raw_lines[end].split("|")[1].split(',')[0]
    lrc_text += f"[{time_start}]{lyric_text[start:end+1]}\n[{time_end}]\n"
print(lrc_text)

lyric_len = len(lyric_text)
for i in range(len(lines_start_chr_idx)-1):
    start = max(0,lines_start_chr_idx[i]-1)
    end = min(lyric_len-1, lines_end_chr_idx[i]+1)
    time_start = raw_lines[start].split("|")[1].split(',')[0]
    time_end = raw_lines[end].split("|")[1].split(',')[0]
    lrc_text += f"[{time_start}]{lyric_text[start:end+1]}\n[{time_end}]\n"
print(lrc_text)