import torch.nn as nn


class LocalDiscriminator(nn.Module):
    def __init__(self, emb_size, seq_len, dropout):
        super().__init__()
        self.dropout = dropout
        self.emb_size = emb_size
        self.seq_len = seq_len - 1
        self.hidden_size = 64
        self.num_class = 4

        self.replace_discriminator = nn.Sequential(
            nn.Dropout(self.dropout, inplace=True),
            nn.Linear(self.emb_size, self.hidden_size, bias=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.num_class, bias=True),
            nn.Softmax(dim=-1)
        )

    def forward(self, token_vec):
        token_predict = self.replace_discriminator(token_vec)
        return token_predict


class GlobalDiscriminator(nn.Module):
    def __init__(self, dropout, emb_size, seq_len):
        super().__init__()
        self.dropout = dropout
        self.emb_size = emb_size
        self.seq_len = seq_len - 1
        self.hidden_size = 32
        self.num_class = 2

        self.fake_discriminator = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size, bias=True),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout, inplace=True),
            nn.Linear(self.hidden_size, self.num_class, bias=True),
            nn.Softmax(dim=-1)
        )

    def forward(self, sent_vec):
        sample_predict = self.fake_discriminator(sent_vec)
        return sample_predict
