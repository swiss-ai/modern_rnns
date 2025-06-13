# Modern RNNS

Codebase for the development of scalable RNNs like DeltaNet and similar.

## Introduction

This codebase was initially developed for the [AI Center Projects in Machine Learning](https://ai.ethz.ch/education/courses/projects-ml-research.html) class at ETH. Our goal was to evaluate the associative memory capacity of sequence decoders with different internal mechanisms — replacing standard attention with linear and recurrent layers, inspired by RNNs.  We test how well these models generalize to longer sequences on structured tasks that require long-range recall.

## Models

We implemented the following models:
- GPT
- Linear Transformer
- LSTM
- quasi-LSTM
- Linear Recurrent Unit (LRU)
- Delta Net
- Mamba

## Datasets

We implemented following datasets. For training, all sequences are randomly generated.

**Bit Parity:**

Tracks the bit parity of a sequence. Outputs a 1 if the count of 1s seen so far is odd.

The lengths of the training and evaluation sequences are configurable.

Example:
```
Input:  0010100
Output: 0011000
```

**Controlled Bit Parity:**
This is the same as Bit Parity, but you can configure a fixed number of ones in the training sequences.

**Dyck:**

Checks for balanced parentheses. Outputs a 1 if the sequence is completely balanced.

The lengths of the training and evaluation sequences, the number of different types of parentheses, and the max nesting depth are configurable.

Example:
```
Input:  {[()]()}[]()
Output: 000000010101
```

**MQAR:**

Multiple Query Associative Retrieval. You give the model a series of Key-Value pairs and then query for a key, and it should return the corresponding value.

The number of unique keys and values, whether to reuse keys, and whether to query for all the keys at the end, are configurable.

**Important:**

The code has been implemented, but our models could never learn well with more than 3 unique keys, although it should be a simple problem and easy to memorize. Despite extensive debugging, we did not figure out why not.

Example:
```
# In reality all these are one-hot encoded.
Input:  c=3, d=1, b=2, c?
Output: 0000 0000 0000 03
```

## How to run

### Running Individual Models:

Under projects, find the model you want to run. Each model has a choice of configurations that are in a "configs.py" file in the same directory. The configuration name must be passed as the first argument, and then you can specify additional arguments.

Example Commands:
```
# See the list of valid configurations
# To run other models, run python projects/lru/mainlru.py, etc.
python projects/lstm/mainlstm.py --help

# See additional configuration options
python projects/lstm/mainlstm.py [config_name] --help

# Run the bit_parity dataset
# Dataset choices: ["bit_parity", "dyck", "mqar"]
PYTHONPATH=. torchrun --standalone projects/lstm/mainlstm.py default --dataset bit_parity

# Use different sequence lengths. The [64,128] is a min and max sequence length. 
PYTHONPATH=. torchrun --standalone projects/lstm/mainlstm.py default --dataset bit_parity --max_seq_len 128 --train_seq_len 16,16 --eval_seq_len 64,128
```

### Running All Models

There is a notebook ```experiment.ipynb``` where you can choose which models and datasets to run, and then it will run them in bulk and upload all the results to Wandb.