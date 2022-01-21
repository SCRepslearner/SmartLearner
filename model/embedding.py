import torch.nn
import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SolEmbedding(nn.Module):
    def __init__(self, type_vocab_size, value_vocab_size, embed_size, pad_idx, dropout):
        super().__init__()
        self.type_emb = torch.nn.Embedding(num_embeddings=type_vocab_size, embedding_dim=embed_size, padding_idx=pad_idx)
        self.value_emb = torch.nn.Embedding(num_embeddings=value_vocab_size, embedding_dim=embed_size, padding_idx=pad_idx)
        # self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, t, v):
        x = self.type_emb(t) + self.value_emb(v)
        return self.dropout(x)
