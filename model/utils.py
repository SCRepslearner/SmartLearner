import torch.nn as nn
import torch
import math
import torch.nn.functional as F


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def pixel_up_sample(x, H, W):
    B, N, C = x.size()
    # batch_size, seq_len, emb_size = x.size()
    assert N == H * W
    # batch_size * emb_size * seq_len
    x = x.permute(0, 2, 1)
    # 2 * emb_size * H * W
    x = x.view(-1, C, H, W)
    # 2 * (emb_size/(factor**2)) * (H*factor) * (W*factor)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    # batch_size *
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Restorer(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        self.layer = nn.Linear(d_model, vocab)

    def forward(self, emb):
        x = self.layer(emb)
        x = F.log_softmax(x, dim=-1)
        x = torch.max(x, dim=2)
        return x.indices


