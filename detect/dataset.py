import os
import random
import pickle
from tqdm import tqdm
from utils.structure_processing import StructureProcessing


class DetectData:
    def __init__(self, source_dir, output_dir, type_vocab, value_vocab, type_vocab_dir, value_vocab_dir, logger,
                 save=True):
        self.source_dir = source_dir
        self.faulty_data = []
        self.test_data = []
        self.save_file = save
        self.logger = logger
        self.output_dir = output_dir
        self.type_vocab = type_vocab
        self.value_vocab = value_vocab
        self.type_vocab_dir = type_vocab_dir
        self.value_vocab_dir = value_vocab_dir
        self.fault_name = "fault_data.pkl"
        self.test_name = "test_data.pkl"
        # the label of valid contract is 0
        self.label_dict = {'A2': 1, 'A6': 2, 'A10': 3, 'A16': 4, 'B1': 5, 'B4': 6, 'B5': 7, 'B7': 8}
        self.load_vocab()

    def load_vocab(self):
        type_vocab_path = os.path.join(self.type_vocab_dir)
        value_vocab_path = os.path.join(self.value_vocab_dir)
        self.type_vocab = self.type_vocab.load_vocab(type_vocab_path)
        self.value_vocab = self.value_vocab.load_vocab(value_vocab_path)

    def load_data(self):
        bug_path = os.path.join(self.output_dir, self.fault_name)
        test_path = os.path.join(self.output_dir, self.test_name)
        if os.path.exists(bug_path):
            with open(bug_path, 'rb') as f:
                self.faulty_data = pickle.load(f)

            with open(test_path, 'rb') as f:
                self.test_data = pickle.load(f)

            random.shuffle(self.faulty_data)
            random.shuffle(self.test_data)

    def prepare_data(self, is_random=False):
        # buggy contract
        database_size = 0
        buggy_size = 0
        for root, dirs, files in os.walk(os.path.join(self.source_dir, "buggy")):
            label = os.path.basename(root)
            if label in self.label_dict:
                count = 0
                test = []
                fault = []
                for file in tqdm(files):
                    if file.endswith('.sol'):
                        file_path = os.path.join(root, file)
                        ast = StructureProcessing(file_path)
                        ast.generate_trees()
                        ast.pre_order_traversal()
                        if count <= len(files) // 2:
                            database_size += 1
                            for sub in ast.samples:
                                sample = {'label': self.label_dict[label], 'file_name': file_path, 'type': sub['type'],
                                          'value': sub['value']}
                                fault.append(sample)
                        else:
                            buggy_size += 1
                            for sub in ast.samples:
                                sample = {'label': self.label_dict[label], 'file_name': file_path, 'type': sub['type'],
                                          'value': sub['value']}
                                test.append(sample)
                        count += 1

                self.faulty_data += fault
                self.test_data += test

        if is_random:
            total = self.faulty_data + self.test_data
            n = len(total) // 2
            total = random.shuffle(total)
            self.faulty_data = total[:n]
            self.test_data = total[n:]

        validated_contracts = []
        valid_size = 0
        # add validated contracts
        flag = True
        for root, dirs, files in os.walk(os.path.join(self.source_dir, "validated")):
            if not flag:
                break
            for file in tqdm(files):
                file_path = os.path.join(root, file)
                ast = StructureProcessing(file_path)
                ast.generate_trees()
                ast.pre_order_traversal()

                valid_size += 1
                if valid_size == buggy_size:
                    flag = False
                    break
                for contract in ast.samples:
                    sample = {'label': -1, 'file_name': file_path, 'type': contract['type'], 'value': contract['value']}
                    validated_contracts.append(sample)
        # buggy:test = 1:1
        validated_contracts = validated_contracts[:len(self.test_data)]
        self.test_data += validated_contracts
        random.shuffle(self.faulty_data)
        random.shuffle(self.test_data)

        self.logger.info("Bug embedding matrix size: file number {}, contracts number {}".format(database_size, len(self.faulty_data)))
        self.logger.info("Test-Buggy: file number {}, contracts number {}".format(buggy_size, len(self.test_data)))
        self.logger.info("Test-Valid: file number {}, contracts number {}".format(valid_size, len(validated_contracts)))

        if self.save_file is True:
            data_path = os.path.join(self.output_dir, self.fault_name)
            with open(data_path, "wb") as f:
                pickle.dump(self.faulty_data, f)
            test_path = os.path.join(self.output_dir, self.test_name)
            with open(test_path, "wb") as f:
                pickle.dump(self.test_data, f)
