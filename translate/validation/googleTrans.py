from googletrans import Translator
translator = Translator()

with open("./data/src.txt", "r") as f:
    src_lines = f.readlines()
    
for line in src_lines:
    result = translator.translate(line, dest='en')
    with open("./data/hyp-gg-py.txt", 'a') as f:
        f.write(result.text + '\n')