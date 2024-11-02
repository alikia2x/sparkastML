import re

def replace_spaces_in_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    new_text = re.sub(r' +', ' ', text)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(new_text)

# 调用函数，替换文件中的空格
replace_spaces_in_file('./data/ug_texts1.txt', './data/2.txt')