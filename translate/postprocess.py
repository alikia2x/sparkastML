import os
import json
from pybloom_live import BloomFilter

def read_converted_files(filename):
    """读取converted.txt文件，返回一个包含已处理文件名的集合"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return set(file.read().splitlines())
    return set()

def write_converted_file(filename, file_name):
    """将处理过的文件名写入converted.txt"""
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(file_name + '\n')

def process_json_files(directory, converted_filename):
    """处理指定目录下的所有json文件"""
    converted_files = read_converted_files(converted_filename)
    bloom_filter_chinese = BloomFilter(capacity=1000000, error_rate=0.001)  # 初始化Bloom Filter
    bloom_filter_english = BloomFilter(capacity=1000000, error_rate=0.001)  # 初始化Bloom Filter

    for filename in os.listdir(directory):
        if filename.endswith('.json') and filename not in converted_files:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                segments = data.get('segments', [])

                with open('./result/source-new.txt', 'a', encoding='utf-8') as source_file, \
                     open('./result/target-new.txt', 'a', encoding='utf-8') as target_file:
                    for segment in segments:
                        chinese_text = segment.get('chinese', '').replace('\n', ' ')
                        english_text = segment.get('english', '').replace('\n', ' ')

                        if chinese_text not in bloom_filter_chinese and english_text not in bloom_filter_english:
                            bloom_filter_chinese.add(chinese_text)
                            source_file.write(chinese_text + '\n')
                            bloom_filter_english.add(english_text)
                            target_file.write(english_text + '\n')

            write_converted_file(converted_filename, filename)

if __name__ == "__main__":
    json_directory = './output-new'  # 替换为你的JSON文件目录路径
    converted_filename = './result/converted.txt'

    process_json_files(json_directory, converted_filename)