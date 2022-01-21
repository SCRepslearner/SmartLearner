import math
import torch
import gc

torch.autograd.set_detect_anomaly(True)
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from .loss import token_loss, sample_loss, restore_loss
from .optim_schedule import ScheduledOptim
from model.model import PretrainModel
from .weighted_loss import AutomaticWeightedLoss


class SolTrainer:

    def __init__(self, model, train_data, test_data, type_vocab, value_vocab, output_dir, batch_size, max_len, lr,
                 betas, weight_decay, warmup_steps, use_gpu: bool = False, gpu: str = "0", logger=None):

        self.logger = logger
        self.my_model = model
        self.model = PretrainModel(self.my_model)
        self.use_gpu = use_gpu
        self.gpu = gpu

        if self.use_gpu:
            if self.gpu == "all":
                self.device = torch.device("cuda:0")
                self.model.to(self.device)
                devices_ids = [i for i in range(torch.cuda.device_count())]
                self.logger.info("Using GPUS:{}  for pretraining".format(devices_ids))
                self.model = nn.DataParallel(self.model, device_ids=devices_ids)
            else:
                self.device = torch.device("cuda:" + self.gpu)
                # self.model_old.to(self.device)
                self.model = self.model.cuda()
                self.logger.info("Using GPU:{} for pretraining".format(self.gpu))
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU for pretraining")

        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.max_len = max_len
        self.min_loss = math.inf

        self.token_criterion = token_loss
        self.sample_criterion = sample_loss
        self.restore_criterion = restore_loss
        self.awl = AutomaticWeightedLoss(3)
        self.optimizer = Adam([
            {'params': self.model.parameters(),
             'weight_decay': weight_decay,
             'lr': lr,
             'betas': betas
             }
        ])

        self.optim_schedule = ScheduledOptim(self.optimizer, self.my_model.hidden, n_warmup_steps=warmup_steps)
        self.output_dir = output_dir
        self.type_vocab = type_vocab
        self.value_vocab = value_vocab

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        loss_log = []
        str_code = "train" if train else "test"
        total_loss = 0
        total_local_loss = 0
        total_global_loss = 0
        total_decode_loss = 0
        for batch_data in tqdm(data_loader):

            data = {key: value.to(self.device) for key, value in batch_data.items()}
            sample_predict, token_predict, value_seq_predict, mask = self.model(data['type'], data['value'], test=False)

            local_loss = self.sample_criterion(sample_predict, data['sample_label'])
            global_loss = self.token_criterion(token_predict, data['token_label'])
            decode_loss = self.restore_criterion(value_seq_predict, data['value'])
            weighted_loss = self.awl(local_loss, global_loss, decode_loss)

            if train:
                self.optim_schedule.zero_grad()
                weighted_loss.backward()
                self.optim_schedule.step_and_update_lr()

            local_loss_value = local_loss.item()
            global_loss_value = global_loss.item()
            decode_loss_value = decode_loss.item()
            weighted_loss_value = weighted_loss.item()

            total_local_loss += local_loss_value
            total_global_loss += global_loss_value
            total_decode_loss += decode_loss_value
            total_loss += weighted_loss_value
            loss_log.append([local_loss_value, global_loss_value, decode_loss_value, weighted_loss_value])
            del local_loss, global_loss, decode_loss, weighted_loss
            del sample_predict, token_predict, mask, value_seq_predict

        gc.collect()

        self.logger.info('{} for epoch: {}, total loss: {}, '.format(str_code, epoch + 1, total_loss))

        if total_loss < self.min_loss:
            self.min_loss = total_loss
            model_save_path = self.output_dir + "model"
            self.save(epoch, model_save_path)

    def save(self, epoch, file_path):
        output_path = file_path + ".ep%d" % (epoch + 1)
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        self.logger.info("EP:{} Model Saved on {}".format(epoch + 1, output_path))
