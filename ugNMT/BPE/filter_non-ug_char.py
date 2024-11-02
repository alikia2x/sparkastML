import re

# 读取文件内容
with open('ug.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# 定义正则表达式，保留维吾尔语字母、阿拉伯数字及常见标点符号
# 维吾尔语字母的Unicode范围是U+0600-U+06FF
# 阿拉伯数字 0-9，以及标点符号（。！？,，；:）可以根据需要调整
filtered_data = re.sub(r'[^\u0600-\u06FF0-9.,!?؛:\s]', '', data)

# 将过滤后的数据输出或保存到新的文件中
with open('filtered_ug.txt', 'w', encoding='utf-8') as file:
    file.write(filtered_data)

print("过滤完成，结果已保存到 filtered_ug.txt")
