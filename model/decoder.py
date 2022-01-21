import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoderLayer
from .utils import Restorer


class Decoder(nn.Module):

    def __init__(self, vocab_size, emb_size, seq_len, dropout, n_layers=6, attn_heads=8):
        super().__init__()
        self.hidden = emb_size
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.feed_forward_hidden = self.hidden * 4
        self.max_len = seq_len
        self.vocab_size = vocab_size

        self.mlp = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(1, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, bias=True),
            nn.LayerNorm(128),
            nn.Linear(128, self.max_len),
        )
        # self.decoder = nn.ModuleList(
        #   [TransformerEncoderLayer(self.hidden, self.attn_heads, self.feed_forward_hidden, self.dropout)
        #     for _ in range(self.n_layers)])

        self.norm = nn.LayerNorm(self.hidden)
        self.dense = nn.Linear(self.hidden, self.vocab_size, bias=True)

    def forward(self, x):
        # batch_size * emb_size => batch_size * emb_size * seq_len
        t = x.clone().unsqueeze(-1)
        f = self.mlp(t).transpose(1, 2)
        # batch_size * emb_size *1 => batch_size * seq_len * emb_size
        # x = x.transpose(1, 2)
        # for layer in self.decoder:
        #    x = layer(x, mask)
        # x = self.norm(x)
        # x = torch.max(F.log_softmax(x, dim=-1))
        logits = self.dense(f)
        return logits


