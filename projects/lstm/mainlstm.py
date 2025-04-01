# Copyright 2023 The Languini Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to train models on the languini books dataset.

# Example calls:

## Single GPU:
CUDA_VISIBLE_DEVICES=0 torchrun --standalone languini/projects/lstm/main.py tiny --train_batch_size 16 --debug

## Multi GPU:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=server.example.com --master_port=12300 \
    languini/projects/lstm/main.py tiny \
    --train_batch_size 16 \
    --eval_every 1000 \
    --log_grads_every 1000 \
    --tokens_per_second 36930 \
    --decay_steps 19475 \
    --max_train_steps 19475 \
    --gradient_accumulation_steps 2
"""

import os
import sys
import torch
import argparse
import torch.multiprocessing as mp

# from languini.train_lib import lm_trainer
# from languini.train_lib import lr_schedules
# from languini.common_lib import parallel_utils
# from languini.common_lib import experiment_utils


# from languini.common_lib.parallel_utils import mprint
# from languini.common_lib.parallel_utils import LOCAL_RANK, WORLD_RANK, WORLD_SIZE

# import configs
from modellstm import Model
from trainers.bit_parity_trainer import BitParityTrainer
from datasets.bit_parity_dataset import BitParityDatasetIterator
from modellstm import Model
from modellstm import DEFAULT_CONFIG as LSTM_MODEL_DEFAULT_CONFIG

SEQ_LEN = None
BATCH_SIZE = 8
VOCAB_SIZE = None

def run(device, dataset):

    if (dataset == "bit_parity"):
        datasetIterator = BitParityDatasetIterator
        trainerClass = BitParityTrainer
        MODEL_DEFAULT_CONFIG = LSTM_MODEL_DEFAULT_CONFIG
        VOCAB_SIZE = 2
    else:
        # add the configuration for each new dataset here
        pass

    SEQ_LEN = MODEL_DEFAULT_CONFIG["seq_len"]

    train_ds = datasetIterator(
        batch_size=BATCH_SIZE,
        sequence_length=SEQ_LEN,
        device=device,
    )

    eval_ds = datasetIterator(
        batch_size=BATCH_SIZE,
        sequence_length=SEQ_LEN,
        device=device,
    )

    ## Setup Model
    # torch.manual_seed(c.seed)
    config = MODEL_DEFAULT_CONFIG
    config["device"] = device
    config["seq_len"] = SEQ_LEN
    config["vocab_size"] = VOCAB_SIZE  # number of output classes for the task
    model = Model(config=config)
    model = model.to(device)

    ## Setup Optimiser
    opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-08)


    ## Setup Trainer
    trainer = trainerClass(
        model=model,
        train_loader=train_ds,
        eval_loader=eval_ds,
        optimizer=opt,
        device=device,
    )

    ## Start Experiment
    print("Begin training ... ")
    trainer.train()
    print("Done!")


def main():
    """Runs an experiment using an LSTM model."""

    parser = argparse.ArgumentParser(description="LSTM Experiment")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to use for training",
    )
    parser.add_argument("--dataset", type=str, choices=["bit_parity", "MQAR"], default="bit_parity",
                        help="Choose dataset: 'bit_parity' (default), 'MQAR'.")
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("LSTM Experiment")

    run(device, args.dataset)


if __name__ == "__main__":
    main()