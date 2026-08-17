"""Minimal whitespace tokenizer + vocab, built from the training corpus.
No external tokenizer library or downloaded vocab needed."""

import re
import json

PAD, MASK, UNK, CLS = "<pad>", "<mask>", "<unk>", "<cls>"
SPECIAL_TOKENS = [PAD, MASK, UNK, CLS]


def basic_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"([.,!?'])", r" \1 ", text)
    return text.split()


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1):
        freq = {}
        for t in texts:
            for tok in basic_tokenize(t):
                freq[tok] = freq.get(tok, 0) + 1
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int = 32):
        toks = [CLS] + basic_tokenize(text)
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in toks][:max_len]
        attn = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.vocab[PAD])
            attn.append(0)
        return ids, attn

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab)

    def __len__(self):
        return len(self.vocab)
