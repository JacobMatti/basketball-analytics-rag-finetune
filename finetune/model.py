"""
A small transformer encoder, built from scratch in PyTorch.

This is intentionally tiny (a few hundred thousand parameters) so it
pretrains and fine-tunes in minutes on CPU in a sandboxed environment with
no GPU and no access to download real pretrained weights. It's a genuine,
if miniature, demonstration of the pretrain -> fine-tune workflow: the same
weights are first trained with a masked-language-modeling objective on
unlabeled text, then reused (not re-initialized) as the starting point for
a downstream classification task.
"""

import torch
import torch.nn as nn


class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x  # (batch, seq_len, d_model) contextualized token representations


class MLMHead(nn.Module):
    """Masked-language-model head used only during pretraining."""

    def __init__(self, encoder: TinyTransformerEncoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(encoder.d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.encoder(input_ids, attention_mask)
        return self.proj(hidden)  # (batch, seq_len, vocab_size)


class ClassificationHead(nn.Module):
    """Downstream fine-tuning head: mean-pools token representations and
    classifies the play-by-play event type."""

    def __init__(self, encoder: TinyTransformerEncoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(encoder.d_model, n_classes)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.encoder(input_ids, attention_mask)  # (b, t, d)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        return self.proj(pooled)  # (batch, n_classes)
