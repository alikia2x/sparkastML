from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

def translate_text(text):
    tokenizer.src_lang = "zh"
    encoded_zh = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return result[0]

with open("./data/src.txt", "r") as f:
    src_lines = f.readlines()
    
for line in tqdm(src_lines):
    result = translate_text(line)
    with open("./data/hyp-m2m.txt", 'a') as f:
        f.write(result + '\n')