# -*- coding: utf-8 -*-
import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class KMeansCluster:
    def __init__(self, output_dir, emb_dim, model, use_gpu, logger):
        self.dimension_num = emb_dim
        self.class_num = 0
        self.component_num = 8
        self.output_dir = output_dir
        self.input_dir = ''
        self.vectors = None
        self.model = model
        self.use_gpu = use_gpu
        self.gpu = '0'
        self.logger = logger
        self.vectors = []
        self.labels = []

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

    def inference(self, data_loader):
        vector_path = os.path.join(self.output_dir, "vectors.pt")
        label_path = os.path.join(self.output_dir, "labels.pt")
        if os.path.exists(vector_path) and os.path.exists(label_path):
            weights = torch.load(vector_path)
            labels = torch.load(label_path)
            weights = weights.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        else:
            for data in tqdm(data_loader):
                if self.use_gpu:
                    data['type'] = data['type'].cuda()
                    data['value'] = data['value'].cuda()
                vectors = self.model.forward(data['type'], data['value'], test=True)
                # print(vectors)
                length = len(data['label'])
                for i in range(length):
                    self.vectors.append(vectors[i])
                    self.labels.append(data['label'][i])

            weights = self.vectors[0].unsqueeze(0)
            for v in self.vectors[1:]:
                weights = torch.cat((weights, v.unsqueeze(0)), dim=0)
            labels = torch.Tensor(self.labels)
            torch.save(weights, os.path.join(self.output_dir, "vectors.pt"))
            torch.save(labels, os.path.join(self.output_dir, "labels.pt"))

            weights = weights.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

        self.class_num = len(set(labels))

        print("K-means")

        self.cluster_k_means(labels, weights)

    def cluster_k_means(self, true_label, data):

        clf = KMeans(n_clusters=self.class_num, max_iter=100, init="k-means++", tol=1e-6)

        _ = clf.fit(data)
        # source = list(clf.predict(data))
        predict_label = clf.labels_

        ARI = metrics.adjusted_rand_score(true_label, predict_label)
        print("adjusted_rand_score: ", ARI)

        # FMI = metrics.fowlkes_mallows_score(true_label, predict_label)
        # print("FMI: ", FMI)
        #
        # silhouette = metrics.silhouette_score(data, predict_label)
        # print("silhouette: ", silhouette)
        #
        # CHI = metrics.calinski_harabasz_score(data, predict_label)
        # print("CHI: ", CHI)
