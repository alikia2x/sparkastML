from openai import OpenAI
import argparse
import os
from dotenv import load_dotenv

def translate_text(text, client, model_name, temp):
    messages = [
        {"role": "system", "content": "User will provide some text. You need to translate the text into English and output it WITHOUT ANY ADDITIONAL INFORMATION OR EXPLANATION."},
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temp,
    )

    return response.choices[0].message.content

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to the input file")
parser.add_argument("output", type=str, help="Path to the output file")
args = parser.parse_args()

input_file = args.input
output_file = args.output
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
model = os.getenv("TRANSLATION_MODEL")
temp = float(os.getenv("TRANSLATION_TEMP"))

with open(input_file, "r") as f:
    src_lines = f.readlines()


for line in src_lines:
    result = translate_text(line, client, model, temp)
    with open(output_file, 'a') as f:
        f.write(result + '\n')
