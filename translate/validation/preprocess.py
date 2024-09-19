import json
import random
import argparse
from tqdm import tqdm


# 读取jsonl文件
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


# 随机抽取一定数量的行
def sample_lines(data, sample_size):
    return random.sample(list(data), sample_size)


# 主函数
def main(input_file, sample_size):
    # 读取jsonl文件
    data = read_jsonl(input_file)

    # 随机抽取一定数量的行
    sampled_data = sample_lines(data, sample_size)

    for item in tqdm(sampled_data):
        chinese_text = item["chinese"]
        english_text = item["english"]

        with open("./data/src.txt", 'a') as srcf, open("./data/ref.txt", 'a') as reff:
            srcf.write(chinese_text + '\n')
            reff.write(english_text + '\n')


# 示例调用
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Process a JSONL file by sampling lines and translating text."
    )

    # 添加命令行参数
    parser.add_argument("input", type=str, help="Path to the input JSONL file")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of lines to sample (default: 100)",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.input, args.sample_size)
