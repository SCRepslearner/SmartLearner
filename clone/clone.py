import collections

import numpy as np
import torch
import os
import pickle
from torch import nn
from model.model import PretrainModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class CloneDetector:
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

    def get_vectors(self, data_loader):
        res = []
        for data in tqdm(data_loader):
            if self.use_gpu:
                data['type'] = data['type'].cuda()
                data['value'] = data['value'].cuda()
            vectors = self.model.forward(data['type'], data['value'], test=True)
            res += vectors

        return res

    def clone_detection(self, data_loader_1, data_loader_2, label, threshold):

        vectors1 = self.get_vectors(data_loader_1)
        vectors2 = self.get_vectors(data_loader_2)
        length = len(vectors1)
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(length):
            dis = torch.dist(vectors1[i], vectors2[i], p=2)
            if dis >= threshold:
                if label[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if label[i] == 1:
                    FN += 1
                else:
                    TN += 1

        print(TP, FP, TN, FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print("P:", precision, "R:", recall, "F1-Score:", f1)
