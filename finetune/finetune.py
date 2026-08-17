"""
Fine-tunes the pretrained encoder on a downstream task: classifying a
play-by-play event into one of 8 categories (MADE_SHOT, TURNOVER, etc).

Runs two conditions to demonstrate that pretraining actually helps, not
just that the pipeline runs:

  1. "Pretrained": ClassificationHead built on top of the encoder weights
     saved by pretrain.py (loaded, not re-initialized).
  2. "From scratch": identical architecture, but with randomly initialized
     encoder weights -- i.e. skipping the pretraining step entirely.

Both are fine-tuned on the same small labeled dataset for the same number
of epochs, then evaluated on a held-out test set, so the comparison isolates
the effect of pretraining.
"""

import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from finetune.model import TinyTransformerEncoder, ClassificationHead
from finetune.tokenizer import SimpleTokenizer

torch.manual_seed(0)

MAX_LEN = 24
LABELS = ["MADE_SHOT", "MISSED_SHOT", "TURNOVER", "FOUL", "REBOUND", "ASSIST", "BLOCK", "STEAL"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


class ClassificationDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=MAX_LEN):
        self.rows = [json.loads(line) for line in open(path)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids, attn = self.tokenizer.encode(row["text"], self.max_len)
        label = LABEL2IDX[row["label"]]
        return torch.tensor(ids), torch.tensor(attn), torch.tensor(label)


def train_and_eval(use_pretrained: bool, tokenizer, train_loader, test_loader, n_epochs=15):
    encoder = TinyTransformerEncoder(vocab_size=len(tokenizer), max_len=MAX_LEN)
    if use_pretrained:
        encoder.load_state_dict(torch.load("finetune/pretrained_encoder.pt"))

    model = ClassificationHead(encoder, n_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for ids, attn, labels in train_loader:
            optimizer.zero_grad()
            logits = model(ids, attn)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, attn, labels in test_loader:
            logits = model(ids, attn)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    tokenizer = SimpleTokenizer.load("finetune/vocab.json")

    train_ds = ClassificationDataset("finetune/finetune_train.jsonl", tokenizer)
    test_ds = ClassificationDataset("finetune/finetune_test.jsonl", tokenizer)

    print(f"Train examples: {len(train_ds)} | Test examples: {len(test_ds)}")

    seeds = [0, 1, 2, 3, 4]
    pretrained_accs, scratch_accs = [], []

    for seed in seeds:
        torch.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=16)

        acc_p = train_and_eval(True, tokenizer, train_loader, test_loader)
        torch.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        acc_s = train_and_eval(False, tokenizer, train_loader, test_loader)

        pretrained_accs.append(acc_p)
        scratch_accs.append(acc_s)
        print(f"Seed {seed}: pretrained={acc_p:.1%}  from_scratch={acc_s:.1%}")

    mean_p = sum(pretrained_accs) / len(pretrained_accs)
    mean_s = sum(scratch_accs) / len(scratch_accs)

    print("\n" + "=" * 60)
    print(f"Mean test accuracy over {len(seeds)} seeds, fine-tuned from pretrained: {mean_p:.1%}")
    print(f"Mean test accuracy over {len(seeds)} seeds, trained from scratch:       {mean_s:.1%}")
    print(f"Pretraining advantage: {(mean_p - mean_s) * 100:+.1f} percentage points")
    print("=" * 60)


if __name__ == "__main__":
    main()
