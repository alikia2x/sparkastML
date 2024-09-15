import os
import json

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

    for filename in os.listdir(directory):
        if filename.endswith('.json') and filename not in converted_files:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                segments = data.get('segments', [])

                with open('./result/source.txt', 'a', encoding='utf-8') as source_file, \
                     open('./result/target.txt', 'a', encoding='utf-8') as target_file:
                    for segment in segments:
                        chinese_text = segment.get('chinese', '').replace('\n', ' ')
                        english_text = segment.get('english', '').replace('\n', ' ')

                        source_file.write(chinese_text + '\n')
                        target_file.write(english_text + '\n')

            write_converted_file(converted_filename, filename)

if __name__ == "__main__":
    json_directory = './output'  # 替换为你的JSON文件目录路径
    converted_filename = './result/converted.txt'

    process_json_files(json_directory, converted_filename)