import pandas as pd

# 定义文件路径
source_files = ['./result/source.txt', './result/source-new.txt']
target_files = ['./result/target.txt', './result/target-new.txt']

# 读取source和target文件内容
source_data = []
target_data = []

for file in source_files:
    with open(file, 'r', encoding='utf-8') as f:
        source_data.extend(f.readlines())

for file in target_files:
    with open(file, 'r', encoding='utf-8') as f:
        target_data.extend(f.readlines())

# 确保source和target行数一致
if len(source_data) != len(target_data):
    print("Warning: The number of lines in source and target files do not match.")

# 创建DataFrame
df = pd.DataFrame({
    'zh': [line.strip() for line in source_data],  # 去掉每行的换行符
    'en': [line.strip() for line in target_data]   # 去掉每行的换行符
})


df.to_csv('./result/data.csv', index=False, encoding='utf-8')
