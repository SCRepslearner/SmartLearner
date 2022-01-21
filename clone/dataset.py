import json
import os
import pickle
from tqdm import tqdm
from utils.structure_processing import StructureProcessing


class CloneData:
    def __init__(self, input_dir, output_dir, type_vocab_dir,
                 value_vocab_dir, type_vocab, value_vocab, logger, save=True):
        self.input_dir = input_dir
        self.type_vocab = type_vocab
        self.value_vocab = value_vocab
        self.logger = logger
        self.output_dir = output_dir
        self.type_vocab_path = type_vocab_dir
        self.value_vocab_path = value_vocab_dir
        self.save_file = save
        self.col1 = []
        self.col2 = []
        self.label = []
        self.load_vocab()

    def load_data(self):
        col1_path = os.path.join(self.output_dir, "col1.pkl")
        col2_path = os.path.join(self.output_dir, "col2.pkl")
        label_path = os.path.join(self.output_dir, "label.pkl")

        if os.path.exists(col1_path):
            with open(col1_path, 'rb') as f:
                self.col1 = pickle.load(f)
            with open(col2_path, 'rb') as f:
                self.col2 = pickle.load(f)
            with open(label_path, 'rb') as f:
                self.label = pickle.load(f)

        assert len(self.col1) == len(self.col2) == len(self.label)
        print(len(self.label))

    def load_vocab(self):
        self.type_vocab = self.type_vocab.load_vocab(self.type_vocab_path)
        self.value_vocab = self.value_vocab.load_vocab(self.value_vocab_path)

    def prepare(self):

        sample_set_1 = []
        sample_set_2 = []

        for root, dirs, files in tqdm(os.walk(self.input_dir)):
            if len(files) == 0:
                continue
            try:
                path_1 = os.path.join(root, files[0])
                path_2 = os.path.join(root, files[1])

                ast1 = StructureProcessing(path_1)
                ast1.generate_trees()
                ast1.pre_order_traversal()
                ast2 = StructureProcessing(path_2)
                ast2.generate_trees()
                ast2.pre_order_traversal()

                contract1 = ast1.samples[0]
                contract2 = ast2.samples[0]

                file_name = os.path.basename(root)
                sample1 = {'label': 1, 'file_name': file_name, 'type': contract1['type'], 'value': contract1['value']}
                sample2 = {'label': 2, 'file_name': file_name, 'type': contract2['type'], 'value': contract2['value']}

                sample_set_1.append(sample1)
                sample_set_2.append(sample2)
            except:
                print("[ERROR]",root)

        assert len(sample_set_1) == len(sample_set_2)

        length = int(len(sample_set_1)/2)
        sample_set_2_random = sample_set_2[length:] + sample_set_2[:length]

        self.col1 = sample_set_1 * 2
        self.col2 = sample_set_2 + sample_set_2_random
        # 1 true clone pairs  0 false clone pairs
        self.label = [1] * len(sample_set_1) + [0] * len(sample_set_1)

        with open(os.path.join(self.output_dir, "col1.pkl"), 'wb') as f:
            pickle.dump(self.col1, f)
        with open(os.path.join(self.output_dir, "col2.pkl"), 'wb') as f:
            pickle.dump(self.col2, f)
        with open(os.path.join(self.output_dir, "label.pkl"), 'wb') as f:
            pickle.dump(self.label, f)



