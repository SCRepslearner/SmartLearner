# SCRL
self-supervised learning of smart contract representation

## Introduction
Learning smart contract representations can greatly facilitate the
development of smart contracts in many tasks such as bug and
clone detection. Existing approaches for learning program representations are difficult to apply to smart contracts which have insufficient data and significant homogenization. To overcome these
challenges, here, we propose SRCL, a novel, self-supervised
approach for learning smart contract representations. Unlike existing supervised methods, which are tied on task-oriented data labels, SRCL leverages large-scale unlabeled data by self-supervised
learning of both local and global information of smart contracts. It
automatically extracts structural sequences from abstract structure
trees (ASTs). Then, two discriminators (local and global) are designed to guide the Transformer encoder to learn local and global
semantic features of smart contracts. 

This repository includes the source code for the paper '
Self-Supervised Learning of Smart Contract Representations'.

## Dependency
```
numpy==1.18.1
torch==1.6.0
tqdm==4.46.0
solidity_parser==0.0.7
treelib==1.6.1
sklearn==0.24.0
```

## Dataset
### train dataset: 
- train dataset could be downloaded from [here](https://drive.google.com/file/d/1dwdKRrX9IkvAF41YlY6expfy_9jgK2vI/view?usp=sharing)

### test dataset

- bug detection
    - [Awesome Buggy ERC20 Tokens](https://github.com/sec-bit/awesome-buggy-erc20-tokens)
    - [OpenZeppelin](https://github.com/OpenZeppelin/)
-  clone detection
    clone detection dataset could be downloaded from [here](https://drive.google.com/file/d/1_EbytiKHzJB6fHFCb-ELjvn9u_SMtdwn/view?usp=sharing)
-  code clustering
    code clustering dataset could be downloaded from [here](https://drive.google.com/file/d/1ni9xyVH6vglFMIoINYYNTirU6KjWK6NJ/view?usp=sharing)

Download the dataset and specify the file location in ``config.py``.

## Usage
### Train
To train our model:
```
python main.py --train
```
### Evaluation
To evaluate our model:
```
python main.py --eval taskname [detect/clone/cluster]
```


## Reference
[1] Liebel, L., & Körner, M. (2018). Auxiliary tasks in multi-task learning. arXiv preprint arXiv:1805.06334.

[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
