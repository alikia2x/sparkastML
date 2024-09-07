import json, random
from torch.utils.data import Dataset

max_dataset_size = 220000

class TRANS19(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        with open(data_file, "rt", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        # 生成不重复的随机行号列表
        random_line_numbers = random.sample(
            range(total_lines), min(max_dataset_size, total_lines)
        )
        random_line_numbers.sort()  # 排序以便按顺序读取文件

        Data = []
        current_line_number = 0

        with open(data_file, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if current_line_number >= len(random_line_numbers):
                    break
                if idx == random_line_numbers[current_line_number]:
                    try:
                        sample = json.loads(line.strip())
                        Data.append(sample)
                    except json.JSONDecodeError:
                        print(f"Error decoding line {idx}")
                    current_line_number += 1

        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]