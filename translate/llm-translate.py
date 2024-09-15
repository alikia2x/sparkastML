import os
from dotenv import load_dotenv
import json
import threading
from openai import OpenAI
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

system_prompt = """
The user will provide some text. Please parse the text into segments, each segment contains 1 to 5 sentences. Translate each sentence into the corresponding language. If the input is in Chinese, return the English translation, and vice versa.

IMPORTANT:
1. Segment should not be too long, each segment should be under 100 English words or 180 Chinese characters.
2. For segments or sentences that appear multiple times in the original text, they are only output **once** in the returned translation.
3. **For content with obvious semantic differences, such as different components on a web page, no matter how short it is, it should be divided into a separate segment.**
4. **Information such as web page headers, footers, and other fixed text, such as copyright notices, website or company names, and conventional link text (such as "About Us", "Privacy Policy", etc.) will be **ignored and not translated**

EXAMPLE INPUT: 
法律之前人人平等，并有权享受法律的平等保护，不受任何歧视。人人有权享受平等保护，以免受违反本宣言的任何歧视行为以及煽动这种歧视的任何行为之害。

EXAMPLE JSON OUTPUT:
{
    "segments": [
        {"chinese": "法律之前人人平等，并有权享受法律的平等保护，不受任何歧视。", "english": "All are equal before the law and are entitled without any discrimination to equal protection of the law."},
        {"chinese": "人人有权享受平等保护，以免受违反本宣言的任何歧视行为以及煽动这种歧视的任何行为之害。", "english": "All are entitled to equal protection against any discrimination in violation of this Declaration and against any incitement to such discrimination."}
    ]
}
"""

def translate_text(text):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={'type': 'json_object'}
    )
    
    return json.loads(response.choices[0].message.content)

def process_file(input_file, output_dir):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        translation = translate_text(text)
        
        output_path = os.path.join(output_dir, Path(input_file).stem + ".json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translation, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully translated and saved to {output_path}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def batch_process(input_dir, output_dir, num_threads=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    
    output_stems = {Path(f).stem for f in output_files}
    
    files = [os.path.join(input_dir, f) for f in input_files if Path(f).stem not in output_stems]
    
    threads = []
    for file in files:
        thread = threading.Thread(target=process_file, args=(file, output_dir))
        threads.append(thread)
        thread.start()
        
        if len(threads) >= num_threads:
            for t in threads:
                t.join()
            threads = []
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    input_dir = "./source"
    output_dir = "./output"
    batch_process(input_dir, output_dir, num_threads=64)