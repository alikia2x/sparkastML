from transformers import AutoModel
from numpy.linalg import norm
import sys
import random
import json
from tqdm import tqdm

# Define the cosine similarity function
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

# Load the model and tokenizer
model_name = 'jinaai/jina-embeddings-v2-base-zh'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Check if the correct number of command-line arguments are provided
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: python script.py <file_path> <output_file_path> [num_samples]")
    sys.exit(1)

# Define file paths from command-line arguments
file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Define the number of samples to randomly select
num_samples = int(sys.argv[3]) if len(sys.argv) == 4 else 100

# Get the total number of lines in the files without loading them fully
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

total_lines = count_lines(file_path)

# Select random sample indices without loading entire files
selected_indices = sorted(random.sample(range(total_lines), num_samples))

# Function to get all sampled lines from the file
def get_lines(file_path, line_numbers):
    result = []
    max_i = max(line_numbers)
    j=0
    next_i = line_numbers[j]
    len_line_numbers = len(line_numbers)
    with open(file_path, 'r', encoding='utf-8') as f:
        for current_line, line in tqdm(enumerate(f)):
            if current_line < next_i:
                continue
            result.append(line.strip())
            j+=1
            if current_line >= max_i or j >= len_line_numbers:
                return result
            next_i = line_numbers[j]
            
    return result

lines = get_lines(file_path, selected_indices)

# Open output file for writing
with open(output_file_path, 'w', encoding='utf-8') as output_file, open("1.txt", 'w', encoding='utf-8') as lf:
    for i, idx in tqdm(enumerate(selected_indices)):
        # Get the corresponding lines from both files
        line = lines[i]
        data = json.loads(line)
        chn = data["chinese"]
        eng = data["english"]
        lf.write(str(idx)+'\n')

        embeddings = model.encode([chn, eng])
        similarity = cos_sim(embeddings[0], embeddings[1])

        # Write the similarity to the output file
        output_file.write(f"{similarity}\n")

print(f"Similarity calculation completed. Results saved to {output_file_path}")
