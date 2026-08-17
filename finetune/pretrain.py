"""
Pretrains the TinyTransformerEncoder with a masked-language-modeling (MLM)
objective on the unlabeled play-by-play corpus -- the same idea BERT-style
foundation models use, just at a scale that trains on CPU in this sandbox.

Output: saved encoder weights + tokenizer vocab, used as the starting point
for finetune.py.
"""

import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from finetune.model import TinyTransformerEncoder, MLMHead
from finetune.tokenizer import SimpleTokenizer, PAD, MASK

random.seed(0)
torch.manual_seed(0)

MAX_LEN = 24
MASK_PROB = 0.15


class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer: SimpleTokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_id = tokenizer.vocab[MASK]
        self.pad_id = tokenizer.vocab[PAD]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, attn = self.tokenizer.encode(self.texts[idx], self.max_len)
        ids = torch.tensor(ids)
        attn = torch.tensor(attn)
        labels = ids.clone()

        maskable = (attn == 1) & (ids != self.tokenizer.vocab.get("<cls>", -1))
        rand = torch.rand(ids.shape)
        mask_positions = maskable & (rand < MASK_PROB)
        labels[~mask_positions] = -100  # ignored in loss
        ids = ids.clone()
        ids[mask_positions] = self.mask_id
        return ids, attn, labels


def load_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def main():
    texts = load_texts("finetune/pretrain_corpus.jsonl")
    tokenizer = SimpleTokenizer.build(texts)
    tokenizer.save("finetune/vocab.json")
    print(f"Vocab size: {len(tokenizer)}")

    dataset = MLMDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    encoder = TinyTransformerEncoder(vocab_size=len(tokenizer), max_len=MAX_LEN)
    model = MLMHead(encoder, vocab_size=len(tokenizer))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    n_epochs = 25
    for epoch in range(n_epochs):
        total_loss, n_batches = 0.0, 0
        for ids, attn, labels in loader:
            optimizer.zero_grad()
            logits = model(ids, attn)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch + 1}/{n_epochs}  MLM loss: {total_loss / n_batches:.4f}")

    torch.save(encoder.state_dict(), "finetune/pretrained_encoder.pt")
    print("Saved pretrained encoder to finetune/pretrained_encoder.pt")


if __name__ == "__main__":
    main()
