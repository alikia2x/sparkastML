import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

def get_AI_response(text, client, model_name, temp):
    messages = [
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temp,
    )

    return response.choices[0].message.content

def get_Examples(df, row, client, model_name, temp):
    exp = df["Example"][row]
    cds = df["Can-do statement"][row]
    gdw = df["guideword"][row]
    lvl = df["Level"][row]
    cat = df["SuperCategory"][row] + '/' + df["SubCategory"][row]
    prompt = \
f'''Generate 10 example sentences based on the following instructions. 
Pay close attention to the 'Can-do Statement' and ensure all generated sentences adhere strictly to it. 
Provide only the sentences without any additional formatting or markdown. 
Output the sentences in plain text, one sentence per line, and do not contain empty line.
INSTRUCTION
Level: {lvl}
Guideword: {gdw}
Can-do Statement: {cds}
Category: {cat}
Example Sentences: 
{exp}
'''
    return get_AI_response(prompt, client, model_name, temp)

def process_chunk(df, chunk, client, model, temp):
    results = []
    for row in chunk:
        exps = get_Examples(df, row, client, model, temp)
        results.append(exps)
    return results

input_file = './EGP.csv'
df = pd.read_csv(input_file)
newdf = df.copy()
model = os.getenv("TRANSLATION_MODEL")
temp = float(os.getenv("TRANSLATION_TEMP"))

chunk_size = 64
total_rows = len(df.index)
num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division

with tqdm(total=total_rows) as pbar:
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk = range(start, end)

        with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
            futures = {executor.submit(get_Examples, df, row, client, model, temp): row for row in chunk}  # 将 row 与 future 绑定
            for future in as_completed(futures):
                row = futures[future]  # 获取对应的行号
                result = future.result()  # 获取 AI 返回的结果
                newdf.at[row, "Example"] = result  # 更新到正确的行

        pbar.update(len(chunk))
        newdf.to_csv("output.csv", index=False)

newdf.to_csv("EGP_Derivied.csv", index=False)