import os

OUTPUT_ROOT = "./output"
INPUT_ROOT = 'D:\\Filez\\DownLoad\\MyDataset'


config = {
    # log
    "log_path": OUTPUT_ROOT + "/sol.log",
    "type_vocab_dir": OUTPUT_ROOT + "/type.vocab",
    "value_vocab_dir": OUTPUT_ROOT + '/value.vocab',

    "train_data": INPUT_ROOT + "/train/",
    "train_output": OUTPUT_ROOT + "/train/",

    "detect_data": INPUT_ROOT + "/detect/",
    "detect_output": OUTPUT_ROOT + "/detect/",

    "cluster_data": INPUT_ROOT + "/cluster/",
    "cluster_output": OUTPUT_ROOT + "/cluster/",

    "clone_data": INPUT_ROOT + "/clone/",
    "clone_output": OUTPUT_ROOT + "/clone/",

    "model_dir": OUTPUT_ROOT + "/model/",

    # vocab_related_parameters
    "value_min_count": 200,
    "type_min_count": 1,
    "emb_size": 256,

    # model_related_parameters
    "max_len": 256,
    "hidden_size": 768,
    "n_layers": 6,
    "attn_heads": 8,
    "dropout": 0.5,

    # training parameters
    "batch_size": 64,
    "learning_rate": 1e-3,

    "gpu": 0,
    "use_gpu": False
}

if not os.path.exists(OUTPUT_ROOT):
    os.mkdir(OUTPUT_ROOT)

if not os.path.exists(config['train_output']):
    os.mkdir(config['train_output'])

if not os.path.exists(config['detect_output']):
    os.mkdir(config['detect_output'])

if not os.path.exists(config['clone_output']):
    os.mkdir(config['clone_output'])

if not os.path.exists(config['cluster_output']):
    os.mkdir(config['cluster_output'])

if not os.path.exists(config['model_dir']):
    os.mkdir(config['model_dir'])