import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, type_vocab, value_vocab, data, max_len):
        self.type_vocab = type_vocab
        self.value_vocab = value_vocab
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        type_seq = sample['type']
        value_seq = sample['value']

        if len(type_seq) > self.max_len:
            type_seq = type_seq[:self.max_len]
            value_seq = value_seq[:self.max_len]

        else:
            type_seq = type_seq + [self.type_vocab.pad_token] * (self.max_len - len(type_seq))
            value_seq = value_seq + [self.value_vocab.pad_token] * (self.max_len - len(value_seq))

        type_ids = self.type_vocab.convert_to_ids(type_seq)
        type_tensor = torch.tensor(type_ids)
        value_ids = self.value_vocab.convert_to_ids(value_seq)
        value_tensor = torch.tensor(value_ids)

        output = {"label": sample['label'],
                  "file": sample['file_name'],
                  "type": type_tensor,
                  "value": value_tensor}

        return output
