import warnings
import logging
import argparse
import torch
import clone.dataset as clone_data
import detect.dataset as bug_data
import cluster.dataset as cluster_data
from train.dataset import TrainDataSet
from data.dataset import BaseDataset as Dataset
from config import config
from clone.clone import CloneDetector
from cluster.cluster_kmeans import KMeansCluster
from train.train import SolTrainer
from detect.detect import BugDetector
from utils.vocab import Vocab
from model.model import Model
from model.encoder import Encoder
from model.decoder import Decoder
from model.discriminator import GlobalDiscriminator, LocalDiscriminator

from torch.utils.data import DataLoader

torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('SCRL: Smart Contract Representation Learner')

    parser.add_argument('--prepare', action='store_true', help='prepare the training data')
    parser.add_argument('--train', action='store_true', help='train SCRL')
    parser.add_argument('--eval', choices=['clone', 'detect', 'cluster'],
                        help='evaluate model in different downstream tasks')
    parser.add_argument('--gpu', default="0", help='specify the gpu device')
    return parser.parse_args()


def train(conf):
    logger = logging.getLogger("sol")
    logger.info('Preparing training data ...')

    type_vocab = Vocab(lower=True)
    value_vocab = Vocab(lower=True)

    dataset = TrainDataSet(input_dir=conf['train_data'], output_dir=conf['train_output'],
                           type_vocab=type_vocab, value_vocab=value_vocab, max_len=conf['max_len'],
                           logger=logger, save_file=True)

    dataset.data = dataset.load_prepared_samples()
    if not dataset.data:
        dataset.data = dataset.prepare_samples()
    logger.info('sample number: {}'.format(dataset.__len__()))
    logger.info("building vocab ...")
    type_vocab.build(samples=dataset.data, min_cnt=conf['type_min_count'], vocab_path=conf['type_vocab_dir'],
                     logger=logger, char_type='type')
    value_vocab.build(samples=dataset.data, min_cnt=conf['value_min_count'], vocab_path=conf['value_vocab_dir'],
                      logger=logger, char_type='value')
    dataset.type_vocab = type_vocab
    dataset.value_vocab = value_vocab
    conf['type_vocab_size'] = type_vocab.size()
    conf['value_vocab_size'] = value_vocab.size()

    data_loader = DataLoader(dataset=dataset, batch_size=conf['batch_size'], shuffle=True)
    logger = logging.getLogger("sol")

    logger.info('Train solidity presentation...')

    dataset.type_vocab.randomly_init_embeddings(conf['emb_size'])
    dataset.value_vocab.randomly_init_embeddings(conf['emb_size'])

    # initialize the model
    encoder = Encoder(use_gpu=False, type_vocab_size=conf['type_vocab_size'], value_vocab_size=conf['value_vocab_size'],
                      emb_size=conf['emb_size'], pad_idx=0, max_len=conf['max_len'], n_layers=6, attn_heads=8,
                      dropout=0.1)
    local_dis = LocalDiscriminator(emb_size=conf['emb_size'], seq_len=conf['max_len'], dropout=0.1)
    global_dis = GlobalDiscriminator(dropout=0.1, seq_len=conf['max_len'], emb_size=conf['emb_size'])
    decoder = Decoder(vocab_size=conf['value_vocab_size'], emb_size=conf['emb_size'], seq_len=conf['max_len'],
                      dropout=0.3)
    model = Model(use_gpu=conf['use_gpu'], encoder=encoder, decoder=decoder, global_dis=global_dis, local_dis=local_dis)

    # load data
    trainer = SolTrainer(model=model, train_data=data_loader, test_data=None, type_vocab=dataset.type_vocab,
                         value_vocab=dataset.value_vocab, output_dir=conf["model_dir"], batch_size=conf['batch_size'],
                         max_len=conf['max_len'], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=1000,
                         use_gpu=conf['use_gpu'], gpu=conf["gpu"], logger=logger)

    for i in range(5):
        trainer.train(epoch=i)


def bug_detection(conf):
    logger = logging.getLogger("sol")
    logger.info("Preparing Bug Detection Data...")

    logger.info("Loading trained model ...")
    model_path = conf['model_dir'] + "model.ep1"
    model = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    detector = BugDetector(model, use_gpu=conf['use_gpu'], gpu=conf['gpu'], output_dir=conf['detect_output'],
                           logger=logger)
    detector.load_emb()

    if len(detector.faulty_data) == 0:
        type_vocab = Vocab(lower=True)
        value_vocab = Vocab(lower=True)
        data = bug_data.DetectData(source_dir=conf['detect_data'], output_dir=conf['detect_output'],
                                   type_vocab_dir=conf['type_vocab_dir'], value_vocab_dir=conf['value_vocab_dir'],
                                   type_vocab=type_vocab, value_vocab=value_vocab, logger=logger)

        data.load_data()
        if not data.faulty_data:
            data.prepare_data()

        # print(len(data.faulty_data), len(data.test_data))

        test_data = Dataset(type_vocab=data.type_vocab, value_vocab=data.value_vocab,
                            data=data.test_data, max_len=conf['max_len'])
        test_data_loader = DataLoader(dataset=test_data, batch_size=conf['batch_size'], shuffle=True)

        fault_data = Dataset(type_vocab=data.type_vocab, value_vocab=data.value_vocab,
                             data=data.faulty_data, max_len=conf['max_len'])
        fault_data_loader = DataLoader(dataset=fault_data, batch_size=conf['batch_size'], shuffle=True)

        with torch.no_grad():
            logger.info("Calculating fault embeddings...")
            detector.prepare_faulty_emb(data_loader=fault_data_loader)
            logger.info("predicting test embeddings...")
            detector.prepare_test_emb(data_loader=test_data_loader)
        # detector.save_emb()
    detector.predict_by_group_similarity()


def clone_detection(conf):
    logger = logging.getLogger("sol")
    logger.info("Preparing Clone Detection Data...")
    logger.info("Loading trained model...")
    model_path = conf['model_dir'] + "model.ep1"
    model = torch.load(model_path, map_location='cpu')
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    detector = CloneDetector(model, use_gpu=conf['use_gpu'], gpu=conf['gpu'],
                             output_dir=conf['clone_output'], logger=logger)

    type_vocab = Vocab(lower=True)
    value_vocab = Vocab(lower=True)

    data = clone_data.CloneData(input_dir=conf['clone_data'], output_dir=conf['clone_output'],
                                type_vocab_dir=conf['type_vocab_dir'], value_vocab_dir=conf['value_vocab_dir'],
                                type_vocab=type_vocab, value_vocab=value_vocab, logger=logger)

    data.load_data()
    if not data.col1:
        data.prepare()

    col1 = Dataset(type_vocab=data.type_vocab, value_vocab=data.value_vocab, data=data.col1,
                   max_len=conf['max_len'])
    col2 = Dataset(type_vocab=data.type_vocab, value_vocab=data.value_vocab, data=data.col2,
                   max_len=conf['max_len'])
    loader_1 = DataLoader(dataset=col1, batch_size=conf['batch_size'])
    loader_2 = DataLoader(dataset=col2, batch_size=conf['batch_size'])
    label1 = data.label

    with torch.no_grad():
        detector.clone_detection(loader_1, loader_2, label1, threshold=0.9)


def code_cluster(conf):
    logger = logging.getLogger("sol")
    logger.info("Code Clustering...")
    logger.info("Loading Pre-trained model...")
    model_path = conf['model_dir'] + "model.ep1"
    model = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    type_vocab = Vocab(lower=True)
    value_vocab = Vocab(lower=True)

    data = cluster_data.ClusterData(input_dir=conf['cluster_data'],output_dir=conf['cluster_output'], type_vocab_dir=conf['type_vocab_dir'],
                                    value_vocab_dir=conf['value_vocab_dir'], type_vocab=type_vocab,
                                    value_vocab=value_vocab)
    data.load_data()
    if len(data.data) == 0:
        data.prepare()

    dataset = Dataset(type_vocab=data.type_vocab, value_vocab=data.value_vocab,
                      data=data.data, max_len=conf['max_len'])
    data_loader = DataLoader(dataset=dataset, batch_size=conf['batch_size'], shuffle=False)
    cluster = KMeansCluster(output_dir=conf['cluster_output'], emb_dim=conf['emb_size'], model=model,
                            use_gpu=conf['use_gpu'], logger=logger)
    with torch.no_grad():
        cluster.inference(data_loader=data_loader)


def run():
    args = parse_args()
    config["gpu"] = args.gpu
    log_path = config['log_path']

    logger = logging.getLogger("sol")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Running with args: {} ".format(args))

    if args.train:
        train(config)
    if args.eval == 'detect':
        bug_detection(config)
    elif args.eval == 'clone':
        clone_detection(config)
    elif args.eval == 'cluster':
        code_cluster(config)


if __name__ == '__main__':
    run()
