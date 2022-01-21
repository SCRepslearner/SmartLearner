import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

criterion = CrossEntropyLoss()


def token_loss(predict, label):
    if predict.is_cuda:
        label = label.cuda()
    predict = predict.transpose(1, 2)
    loss = criterion(predict, label)
    return loss


def sample_loss(predict, label):
    if predict.is_cuda:
        label = label.cuda()
    loss = criterion(predict, label)
    return loss


def restore_loss(predict, label):
    predict = predict.transpose(1, 2)
    loss = criterion(predict, label)
    return loss
