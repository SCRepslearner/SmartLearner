import torch
import math
import random
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, use_gpu, encoder, decoder, global_dis, local_dis):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.global_dis = global_dis
        self.local_dis = local_dis
        self.hidden = encoder.hidden
        self.use_gpu = use_gpu

    def forward(self, type, value, test=False):

        local_resp, global_resp, mask = self.encoder(type, value, mask=None)

        # test
        if test:
            return global_resp

        # discriminator
        token_predict = self.local_dis(local_resp)
        sample_predict = self.global_dis(global_resp)
        # decoder
        restored_value = self.decoder(global_resp)
        return sample_predict, token_predict, restored_value, mask


class PretrainModel(nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.pretrain_model = model

    def forward(self, type, value, test=False):
        return self.pretrain_model.forward(type, value, test)
