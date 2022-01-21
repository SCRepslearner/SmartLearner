import torch.nn as nn
from .attention import MultiHeadedAttention, SelfAttention, SourceTargetAttention
from .utils import SublayerConnection, PositionwiseFeedForward
from .fnn import FFN


class TransformerEncoderLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()

        # Self Attention Layer
        # query key and value come from previous layer.
        self.self_attn = SelfAttention(attn_heads, hidden, dropout)
        # Source Target Attention Layer
        # query come from encoded space.
        # key and value come from previous self attention layer
        self.st_attn = SourceTargetAttention(attn_heads, hidden, dropout)
        self.ff = FFN(hidden, feed_forward_hidden)

    def forward(self, x, mem, source_mask, target_mask):
        # self attention
        x = self.self_attn(x, target_mask)
        # source target attention
        x = self.st_attn(mem, x, source_mask)
        # pass through feed forward network
        return self.ff(x)
