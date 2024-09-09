import os
from dotenv import load_dotenv
import json
import threading
from openai import OpenAI

load_dotenv()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# 系统提示词
system_prompt = """
The user will provide some text. Please parse the text into segments, each segment contains 1 to 5 sentences. Translate each sentence into the corresponding language. If the input is in Chinese, return the English translation, and vice versa.

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

# 翻译函数
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

# 处理单个文件的函数
def process_file(input_file, output_dir):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        translation = translate_text(text)
        
        output_file = os.path.join(output_dir, os.path.basename(input_file) + '.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translation, f, ensure_ascii=False, indent=4)
        
        print(f"Processed {input_file} and saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# 批量处理目录下的文件
def batch_process(input_dir, output_dir, num_threads=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    threads = []
    for file in files:
        thread = threading.Thread(target=process_file, args=(file, output_dir))
        threads.append(thread)
        thread.start()
        
        # 控制线程数量
        if len(threads) >= num_threads:
            for t in threads:
                t.join()
            threads = []
    
    # 等待剩余线程完成
    for t in threads:
        t.join()

# 主函数
if __name__ == "__main__":
    input_dir = "./source"
    output_dir = "./output"
    batch_process(input_dir, output_dir)