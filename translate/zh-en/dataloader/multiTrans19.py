import json, random
from torch.utils.data import Dataset

max_dataset_size = 82000

class MultiTRANS19(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = []
        file_lines = []

        with open(data_file, "rt", encoding="utf-8") as f:
            file_lines = f.readlines()

        combine_number_list = []
        for _ in range(max_dataset_size):
            num = random.randint(2, 7) 
            combine_number_list.append(num)

        file_lines = random.sample(file_lines, sum(combine_number_list))
        
        total = 0
        for combine_count in combine_number_list:
            num_combination = combine_number_list[combine_count]
            sample = {
                "chinese": "",
                "english": ""
            }
            for line in file_lines[total: total+num_combination]:
                try:
                    line_sample = json.loads(line.strip())
                    sample["chinese"] += line_sample["chinese"]
                    sample["english"] += line_sample["english"]
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {e}")
            
            Data.append(sample)
            total+=num_combination
            
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
