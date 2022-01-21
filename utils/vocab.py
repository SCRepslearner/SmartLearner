import numpy as np
import pickle


class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """

    def __init__(self, filename=None, lower=True):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.start_token = "<start>"
        self.number_token = "<number>"
        self.str_token = "<str>"

        self.initial_tokens = [self.pad_token, self.unk_token, self.start_token, self.number_token, self.str_token]

        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_from_file(filename)

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    def size(self):
        return len(self.id2token)

    def load_from_file(self, file_path):
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add(token)

    def get_id(self, token):
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def convert_to_ids(self, tokens):
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens

    def randomly_init_embeddings(self, embed_dim):
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    def build(self, samples, min_cnt, vocab_path, logger, char_type):
        for sample in samples:
            for token in sample[char_type]:
                self.add(token)

        unfiltered_vocab_size = self.size()
        self.filter_tokens_by_cnt(min_cnt)
        filtered_num = unfiltered_vocab_size - self.size()
        print(char_type + " vocab size:",len(self.id2token))
        logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num, self.size()))
        self.save_vocab(vocab_path)


