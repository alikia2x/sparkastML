import torch
from transformers import AutoModel, AutoTokenizer
from numpy.linalg import norm
import sys
import random
from tqdm import tqdm

# Define the cosine similarity function
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

# Load the model and tokenizer
model_name = 'jinaai/jina-embeddings-v2-base-zh'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Check if the correct number of command-line arguments are provided
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: python script.py <file_a_path> <file_b_path> <output_file_path> [num_samples]")
    sys.exit(1)

# Define file paths from command-line arguments
file_a_path = sys.argv[1]
file_b_path = sys.argv[2]
output_file_path = sys.argv[3]

# Define the number of samples to randomly select
num_samples = int(sys.argv[4]) if len(sys.argv) == 5 else 100

# Get the total number of lines in the files without loading them fully
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

total_lines_a = count_lines(file_a_path)
total_lines_b = count_lines(file_b_path)

# Ensure both files have the same number of lines
if total_lines_a != total_lines_b:
    print("Files must have the same number of lines.")
    sys.exit(1)

# Select random sample indices without loading entire files
selected_indices = sorted(random.sample(range(total_lines_a), num_samples))

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

lines_a = get_lines(file_a_path, selected_indices)
lines_b = get_lines(file_b_path, selected_indices)

# Open output file for writing
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for i, idx in tqdm(enumerate(selected_indices)):
        # Get the corresponding lines from both files
        line_a = lines_a[i]
        line_b = lines_b[i]

        embeddings = model.encode([line_a, line_b])
        similarity = cos_sim(embeddings[0], embeddings[1])

        # Write the similarity to the output file
        output_file.write(f"{similarity}\n")

print(f"Similarity calculation completed. Results saved to {output_file_path}")
