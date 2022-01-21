import collections

import numpy as np
import torch
import os
import pickle
from torch import nn
from model.model import PretrainModel
from tqdm import tqdm


class BugDetector:
    def __init__(self, sub_model, use_gpu, gpu, output_dir, logger=None):
        self.logger = logger
        self.sub_model = sub_model
        self.model = PretrainModel(self.sub_model)
        self.use_gpu = use_gpu
        self.gpu = gpu
        self.output_dir = output_dir
        self.threshold = 0.95

        if self.use_gpu:
            if self.gpu == "all":
                self.device = torch.device("cuda:0")
                self.model.to(self.device)
                devices_ids = [i for i in range(torch.cuda.device_count())]
                self.logger.info("Using GPUS:{} for evaluating".format(devices_ids))
                self.model = nn.DataParallel(self.model, device_ids=devices_ids)
            else:
                self.device = torch.device("cuda:" + self.gpu)
                self.model.to(self.device)
                self.logger.info("Using GPU:{} for evaluating".format(self.gpu))
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU for evaluating")

        self.faulty_data = {}
        self.test_data = {}

    def prepare_faulty_emb(self, data_loader):
        for data in tqdm(data_loader):
            if self.use_gpu:
                data['type'] = data['type'].cuda()
                data['value'] = data['value'].cuda()
            sent_vec = self.model.forward(data['type'], data['value'], test=True)
            length = len(data['label'])
            for i in range(length):
                if data['file'][i] not in self.faulty_data:
                    self.faulty_data[data['file'][i]] = {'label': data['label'][i], 'sub_contracts': [sent_vec[i]]}
                else:
                    self.faulty_data[data['file'][i]]['sub_contracts'].append(sent_vec[i])

    def prepare_test_emb(self, data_loader):
        for data in tqdm(data_loader):
            # data = {key: value.to(self.device) for key, value in batch_data.items()}
            if self.use_gpu:
                data['type'] = data['type'].cuda()
                data['value'] = data['value'].cuda()
            sent_vec = self.model.forward(data['type'], data['value'], test=True)
            length = len(data['label'])
            for i in range(length):
                if data['file'][i] not in self.test_data:
                    self.test_data[data['file'][i]] = {'label': data['label'][i], 'sub_contracts': [sent_vec[i]]}
                else:
                    self.test_data[data['file'][i]]['sub_contracts'].append(sent_vec[i])

    def predict_by_group_similarity(self):

        labels = []

        for idy, fault in self.faulty_data.items():
            labels.append(fault['label'])

        TP, FP, TN, FN = 0, 0, 0, 0
        for idx, test in self.test_data.items():
            all_sim_list = []
            for idy, fault in self.faulty_data.items():
                item_sim = []
                for sub_test in test['sub_contracts']:
                    sub_sim = []
                    for sub_fault in fault['sub_contracts']:
                        temp = torch.dist(sub_test, sub_fault, p=2)
                        sub_sim.append(temp)
                    avg_sim = torch.max(torch.stack(sub_sim))
                    item_sim.append(avg_sim)
                all_sim_list.append(torch.mean(torch.stack(item_sim)))
            max_sim = max(all_sim_list)
            predict_idx = all_sim_list.index(max(all_sim_list))
            predict_label = labels[predict_idx].item()
            true_label = test['label'].item()
            if max_sim > self.threshold:
                if true_label != -1:
                    TP += 1
                else:
                    FP += 1
            else:
                if true_label == -1:
                    TN += 1
                else:
                    FN += 1

        accuracy = TP + TN / (TP + TN + FP + FN)
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if (TP + FN) == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        print(accuracy, precision, recall, f1)

    def save_emb(self):
        fault_path = os.path.join(self.output_dir, "fault_emb.pkl")
        test_path = os.path.join(self.output_dir, "test_emb.pkl")

        with open(fault_path, "wb") as f:
            pickle.dump(self.faulty_data, f)

        with open(test_path, "wb") as f:
            pickle.dump(self.test_data, f)

    def load_emb(self):
        fault_path = os.path.join(self.output_dir, "fault_emb.pkl")
        test_path = os.path.join(self.output_dir, "test_emb.pkl")

        if os.path.exists(fault_path):
            with open(fault_path, 'rb') as f:
                self.faulty_data = pickle.load(f)

            with open(test_path, 'rb') as f:
                self.test_data = pickle.load(f)
