from transformers import AutoModel
from numpy.linalg import norm
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Usage: python filter.py <file_a_path> <file_b_path> <output_file_path>"
)

parser.add_argument("file_a", type=str, help="File No.1")
parser.add_argument("file_b", type=str, help="File No.2")
parser.add_argument("output", type=str, help="Output file")
parser.add_argument(
    "--resume",
    type=int,
    default=-1,
    help="Resume from specified line",
)
args = parser.parse_args()

# Define the cosine similarity function
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

# Load the model and tokenizer
model_name = 'jinaai/jina-embeddings-v2-base-zh'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to('cuda')

# Define file paths from command-line arguments
file_a_path = args.file_a
file_b_path = args.file_b
output_file_path = args.output

resume_from = args.resume
resume = resume_from >= 0
output_file_mode = 'a' if resume else 'w'

# Open files
with open(file_a_path, 'r', encoding='utf-8') as file_a, \
     open(file_b_path, 'r', encoding='utf-8') as file_b, \
     open(output_file_path, output_file_mode, encoding='utf-8') as output_file:
    i=1
    # Read file A and file B line by line
    for line_a, line_b in tqdm(zip(file_a, file_b)):
        if resume and i < resume_from:
            i+=1
            continue
        # Remove trailing newline characters
        line_a = line_a.strip()
        line_b = line_b.strip()
        
        embeddings = model.encode([line_a, line_b])
        similarity = cos_sim(embeddings[0], embeddings[1])
        
        # Write the similarity to the output file
        output_file.write(f"{similarity}\n")
        
        i+=1

print(f"Similarity calculation completed. Results saved to {output_file_path}")