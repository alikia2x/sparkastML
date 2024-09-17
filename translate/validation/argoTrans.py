import subprocess

def translate_text(text):
    command = f'argos-translate --from zh --to en "{text}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

with open("src.txt", "r") as f:
    src_lines = f.readlines()
    
for line in src_lines:
    result = translate_text(line)
    with open("hyp-ag.txt", 'a') as f:
        f.write(result + '\n')