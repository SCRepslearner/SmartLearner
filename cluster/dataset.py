import os
import pickle
from utils.structure_processing import StructureProcessing


class ClusterData:
    def __init__(self, input_dir, output_dir, type_vocab_dir, value_vocab_dir, type_vocab, value_vocab):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.type_vocab_dir = type_vocab_dir
        self.value_vocab_dir = value_vocab_dir
        self.type_vocab = type_vocab
        self.value_vocab = value_vocab
        self.data = []
        self.load_vocab()

    def load_vocab(self):
        self.type_vocab = self.type_vocab.load_vocab(self.type_vocab_dir)
        self.value_vocab = self.value_vocab.load_vocab(self.value_vocab_dir)

    def load_data(self):
        if os.path.exists(os.path.join(self.output_dir, "cluster.pkl")):
            with open(os.path.join(self.output_dir, "cluster.pkl"), 'rb') as f:
                self.data = pickle.load(f)

    def prepare(self):
        label = 0
        for root, dirs, files in os.walk(self.input_dir):
            if root == self.input_dir:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                ast = StructureProcessing(file_path)
                ast.generate_trees()
                ast.pre_order_traversal()
                for sub in ast.samples:
                    sample = {'label': label, 'file_name': file, 'type': sub['type'], 'value': sub['value']}
                    self.data.append(sample)
            label += 1

        print(label, len(self.data))

        with open(os.path.join(self.output_dir, "cluster.pkl"), 'wb') as f:
            pickle.dump(self.data, f)
