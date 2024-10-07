import json
import subprocess
import evaluate
from nltk.tokenize import word_tokenize
from tqdm import tqdm

bleu_cal = evaluate.load("chrf")

def translate_text(text):
    command = f'argos-translate --from zh --to en "{text}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    # 读取数据集
    with open('./data/1.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    translations = []
    references = []
    
    # for entry in tqdm(data):
    #     chinese_sentence = entry['zh']
    #     translated_sentence = translate_text(chinese_sentence)
    #     with open("./data/1-inf.txt", "a") as f:
    #         f.write(translated_sentence + "\n")
    #     translations.append(translated_sentence)
    
    with open("./data/1-inf.txt", 'r') as f:
        translations = f.readlines()

    for entry in data:
        english_sentence = entry['en']
        references.append([english_sentence])
        

    # 计算 BLEU 分数
    bleu = bleu_cal.compute(predictions=translations, references=references)
    print(bleu)

if __name__ == "__main__":
    main()