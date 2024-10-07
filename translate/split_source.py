import os
import re

def split_content(content):
    sentences = re.split(r'[。！？；.!?;]', content)
    segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if (len(current_segment) >= 25 or current_length + sentence_length > 1200):
            segments.append(''.join(current_segment))
            current_segment = []
            current_length = 0
        
        current_segment.append(sentence)
        current_length += sentence_length
    
    if current_segment:
        segments.append(''.join(current_segment))
    
    return segments

def process_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # 只处理文件，跳过目录
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            segments = split_content(content)
            
            if len(segments) > 1:
                # 删除原始文件
                os.remove(file_path)
                
                # 保存分割后的文件
                for i, segment in enumerate(segments):
                    new_filename = f"{filename}_{i+1}"
                    new_file_path = os.path.join(directory, new_filename)
                    
                    with open(new_file_path, 'w', encoding='utf-8') as new_file:
                        new_file.write(segment)
            else:
                print(f"文件 {filename} 不需要分割")

# 指定目录
directory = './source-new'
process_files_in_directory(directory)