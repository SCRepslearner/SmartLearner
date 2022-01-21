import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import SolEmbedding
from .transformer import TransformerEncoderLayer


def conv_and_pool(x, conv):
    feature = F.relu(conv(x)).squeeze(3)
    out = F.max_pool1d(feature, int(feature.size(2))).squeeze(2)
    return out


class Encoder(nn.Module):

    def __init__(self, use_gpu, type_vocab_size, value_vocab_size, emb_size, pad_idx, max_len, n_layers=6,
                 attn_heads=12, dropout=0.1):
        super().__init__()
        self.type_vocab_size = type_vocab_size
        self.value_vocab_size = value_vocab_size
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.hidden = emb_size
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.use_gpu = use_gpu

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = emb_size * 4

        self.embedding = SolEmbedding(type_vocab_size=self.type_vocab_size, value_vocab_size=self.value_vocab_size,
                                      embed_size=self.hidden, dropout=self.dropout, pad_idx=self.pad_idx)

        # multi-layers transformer blocks, deep network
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoderLayer(self.hidden, self.attn_heads, self.feed_forward_hidden, self.dropout)
             for _ in range(self.n_layers)])

        # CNN
        self.kernel_size = [3, 5, 7, 9]
        self.filter_num = int(self.hidden / len(self.kernel_size))
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.filter_num, (k, self.hidden)) for k in self.kernel_size])

    def forward(self, type_seq, value_seq, mask):
        # attention masking for padded token
        if mask is None:
            mask = (value_seq > 0).unsqueeze(1).repeat(1, value_seq.size(1), 1).unsqueeze(1)
        if self.use_gpu:
            mask = mask.cuda()

        # embedding the indexed sequence to sequence of vectors
        local_resp = self.embedding(type_seq, value_seq)

        # running over multiple transformer blocks
        for transformer in self.transformer_encoder:
            local_resp = transformer.forward(local_resp, mask)

        # get global representation by cnn
        fea = local_resp.unsqueeze(1)
        global_resp = torch.cat([conv_and_pool(fea.clone(), conv) for conv in self.convs], 1)
        return local_resp, global_resp, mask
