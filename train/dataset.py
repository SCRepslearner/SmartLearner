import os
import torch
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.structure_processing import StructureProcessing
from utils.data_augmentation import da_operations

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TrainDataSet(Dataset):
    def __init__(self, input_dir, output_dir, type_vocab, value_vocab, max_len, logger, save_file=True):
        self.total_file = 0
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.save_file = save_file
        self.logger = logger
        self.type_vocab = type_vocab
        self.value_vocab = value_vocab
        self.max_len = max_len
        self.data = []

    def __getitem__(self, item):
        # start + seq
        type_seq = self.data[item]['type']
        value_seq = self.data[item]['value']
        token_label = self.data[item]['token_label']
        length = len(type_seq)
        if length > self.max_len:
            type_seq = type_seq[:self.max_len]
            value_seq = value_seq[:self.max_len]
            token_label = token_label[:self.max_len]
        else:
            type_seq = type_seq + [self.type_vocab.pad_token] * (self.max_len - length)
            value_seq = value_seq + [self.value_vocab.pad_token] * (self.max_len - length)
            token_label = token_label + [0] * (self.max_len - length)

        type_ids = self.type_vocab.convert_to_ids(type_seq)
        value_ids = self.value_vocab.convert_to_ids(value_seq)

        output = {"type": type_ids,  # type_seq
                  "value": value_ids,  # api_seq
                  "sample_label": self.data[item]['sample_label'],
                  "token_label": token_label
                  }
        return {key: torch.tensor(value) for key, value in output.items()}

    def __len__(self):
        return len(self.data)

    def load_prepared_samples(self):
        data_path = os.path.join(self.output_dir, "sub_contracts.pkl")
        print(data_path)
        self.logger.info("loading prepared data from {}".format(data_path))
        if not os.path.exists(data_path):
            self.logger.info('provided file path not found')
            return []

        with open(data_path, 'rb') as f:
            samples = pickle.load(f)
        self.logger.info("sample size :{}".format(len(samples)))
        return samples

    def prepare_samples(self):
        """
        traverse the dir load and parse all solidity files
        """
        self.logger.info('traverse the dir, load and parse all solidity files!')
        if not os.path.exists(self.input_dir):
            self.logger.info('provided input_dir:' + self.input_dir + ' not exists!')

        for root, dirs, files in os.walk(self.input_dir):
            for file in tqdm(files):
                self.total_file += 1
                file_path = os.path.join(root, file)
                ast = StructureProcessing(file_path)
                ast.generate_trees()
                ast.pre_order_traversal()
                if len(ast.samples) == 0:
                    print(file)
                    continue
                for sample in ast.samples:
                    sample["name"] = file
                    self.data.append(sample)

        self.logger.info("the number of files:{}".format(self.total_file))
        self.logger.info("sample size: {}".format(len(self.data)))
        print('samples size :', len(self.data))
        # data augmentation
        self.logger.info("data augmentation")
        self.data = da_operations(self.data)
        self.logger.info("augmented sample size:".format(len(self.data)))
        print('augmented samples size :',len(self.data))
        if self.save_file is True:
            data_path = os.path.join(self.output_dir, "sub_contracts.pkl")
            with open(data_path, "wb") as f:
                pickle.dump(self.data, f)
        return self.data
