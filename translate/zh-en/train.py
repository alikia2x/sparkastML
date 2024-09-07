import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from sacrebleu.metrics import BLEU
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader.wikititle import Wikititle

writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_set_size = 95000
valid_set_size = 5000
test_data_size = 0

last_1k_loss = []
kmean_loss = 0.0
total_loss = 0.0
best_bleu = 0.0
step = 0

max_input_length = 128
max_target_length = 128

batch_size = 8
learning_rate = 1e-5
epoch_num = 1

# 检查点文件路径，默认为None
# checkpoint_path = None
checkpoint_path = "./saves/checkpoint_74000.bin"  # 如果要从检查点继续训练，设置此路径


data = Wikititle("./data/wikititles-v3.zh-en.tsv")
train_data, valid_data, test_data = random_split(data, [train_set_size, valid_set_size, test_data_size])

# data = TRANS("./data/translation2019zh/translation2019zh_train.json")
# train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
# test_data = TRANS("./data/translation2019zh/translation2019zh_valid.json")

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

# 如果指定了检查点路径，则从检查点加载模型状态
if checkpoint_path is not None:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    total_loss = checkpoint_data.get("total_loss", 0.0)
    step = checkpoint_data.get("step", 0)
    kmean_loss = total_loss / step
    last_1k_loss = [kmean_loss] * 1000


def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample["chinese"])
        batch_targets.append(sample["english"])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        batch_data["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
            labels
        )
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx + 1 :] = -100
        batch_data["labels"] = labels

    batch_data = {k: v.to(device) for k, v in batch_data.items()}
    return batch_data


train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn
)
valid_dataloader = DataLoader(
    valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn
)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn
)


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss, step):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        del last_1k_loss[0]
        last_1k_loss.append(loss.item())
        kmean_loss = sum(last_1k_loss) / len(last_1k_loss)
        progress_bar.set_description(
            f"loss: {kmean_loss:>7f}"
        )
        progress_bar.update(1)

        step += 1
        writer.add_scalar("Loss", kmean_loss, step)
        writer.add_scalar("Overall Loss", total_loss / step, step)

        if step % 250 == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "total_loss": total_loss,
                "kmean_loss": kmean_loss,
                "step": step,
            }
            torch.save(checkpoint, f"checkpoint_{step}.bin")
    return total_loss, step


bleu = BLEU()


def test_loop(dataloader, model):
    preds, labels = [], []
    model.eval()
    for batch_data in tqdm(dataloader):
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    max_length=max_target_length,
                    no_repeat_ngram_size=3,
                )
                .cpu()
                .numpy()
            )

        label_tokens = batch_data["labels"].cpu().numpy()
        decoded_preds = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        label_tokens = np.where(
            label_tokens != -100, label_tokens, tokenizer.pad_token_id
        )
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f"BLEU: {bleu_score:>0.2f}\n")
    return bleu_score


optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * epoch_num * len(train_dataloader)),
    num_training_steps=epoch_num * len(train_dataloader),
)

for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n {'-'*20}")
    total_loss, step = train_loop(
        train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss, step
    )
    valid_bleu = test_loop(valid_dataloader, model)
    print("saving new weights...\n")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "total_loss": total_loss,
        "kmean_loss": kmean_loss,
        "step": step,
    }
    torch.save(checkpoint, f"step_{step}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin")

print("Done!")
