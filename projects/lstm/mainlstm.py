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

import argparse
import os
import sys

import torch

# from languini.train_lib import lm_trainer
# from languini.train_lib import lr_schedules
# from languini.common_lib import parallel_utils
from common_lib import experiment_utils
from common_lib.parallel_utils import mprint

# from languini.common_lib.parallel_utils import LOCAL_RANK, WORLD_RANK, WORLD_SIZE

import configs
from basic_lstm_model import ModelLSTM 
from datasets.mqar_dataset import MQARDatasetIterator
from trainers.bit_parity_trainer import BitParityTrainer
from trainers.dyck_trainer import DyckTrainer
from datasets.bit_parity_dataset import BitParityDatasetIterator
from datasets.dyck_dataset import DyckDatasetIterator
from trainers.mqar_trainer import MQARTrainer


def run(config, logger):

    if config.dataset == "bit_parity":
        train_ds = BitParityDatasetIterator(
            batch_size=config.train_batch_size,
            sequence_length=config.train_seq_len,
            pad_sequence_length=config.max_seq_len,
            device=config.device,
        )
        eval_ds = BitParityDatasetIterator(
            batch_size=config.eval_batch_size,
            sequence_length=config.eval_seq_len,
            pad_sequence_length=config.max_seq_len,
            device=config.device,
        )

        trainerClass = BitParityTrainer
    elif config.dataset == "dyck":
        train_ds = DyckDatasetIterator(
            batch_size=config.train_batch_size,
            sequence_length=config.train_seq_len,
            pad_sequence_length=config.max_seq_len,
            device=config.device,
            depth=config.depth,
            num_parentheses=config.num_parentheses,
        )
        eval_ds = DyckDatasetIterator(
            batch_size=config.eval_batch_size,
            sequence_length=config.eval_seq_len,
            pad_sequence_length=config.max_seq_len,
            device=config.device,
            depth=config.depth,
            num_parentheses=config.num_parentheses,
        )
        config.num_input_classes = config.num_parentheses * 2 + 1

        trainerClass = DyckTrainer
    elif config.dataset == "mqar":
        train_ds = MQARDatasetIterator(
            batch_size=config.train_batch_size,
            num_pairs=config.train_num_pairs,
            n_keys=config.n_keys,
            n_values=config.n_values,
            pad_num_pairs=config.max_num_pairs,
            unique_keys=config.unique_keys,
            unique_values=config.unique_values,
            all_queries_for_input=config.all_queries_for_input,
            device=config.device,
        )
        eval_ds = MQARDatasetIterator(
            batch_size=config.eval_batch_size,
            num_pairs=config.eval_num_pairs,
            n_keys=config.n_keys,
            n_values=config.n_values,
            pad_num_pairs=config.max_num_pairs,
            unique_keys=config.unique_keys,
            unique_values=config.unique_values,
            all_queries_for_input=config.all_queries_for_input,
            device=config.device,
        )
        config.num_input_classes = max(config.n_keys, config.n_values + 1) + 1
        config.output_size = config.n_values + 1
        config.max_seq_len = max(config.max_num_pairs * 3, config.max_seq_len)

        trainerClass = MQARTrainer 
    else:
        # add the configuration for each new dataset here
        raise RuntimeError(
            f"Dataset {config.dataset} not supported. Please add the configuration for this dataset."
        )

    ## Setup Model
    torch.manual_seed(config.seed)
    model = ModelLSTM(config=config)
    model = model.to(config.device)

    ## Setup Optimiser
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95), eps=1e-08)

    ## Setup Trainer
    trainer = trainerClass(
        config=config,
        model=model,
        train_loader=train_ds,
        eval_loader=eval_ds,
        optimizer=opt,
        device=config.device,
        logger=logger,
    )

    ## Start Experiment
    print("Begin training ... ")
    trainer.train()
    print("Done!")


def main():
    """Runs an experiment using an LSTM model."""
    config_name = experiment_utils.parse_config_name(configs.config_names)
    mprint(f"Loading config: {config_name}")

    # load the config file
    config = configs.load_config(name=config_name)
    config.project_path = os.path.dirname(os.path.abspath(__file__))
    mprint(f"project path: {config.project_path}")

    parser = experiment_utils.create_parser_based_on_config(config)

    # Add custom arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bit_parity", "dyck", "mqar"],
        default="bit_parity",
        help="Choose dataset: 'bit_parity' (default), 'dyck', 'mqar'.",
    )
    args = parser.parse_args(sys.argv[2:])
    config = experiment_utils.update_config_given_args(config, args)
    config.dataset = args.dataset

    if args.device == "gpu":
        if torch.cuda.is_available():
            config.device = torch.device("cuda")
        else:
            mprint("Cuda is not available, using CPU instead.")
            config.device = torch.device("cpu")
    else:
        config.device = torch.device("cpu")

    # Generate experiment name based on config
    configs.add_exp_name(config)
    mprint(f"experiment name: {config.exp_name}")

    # Create the log folder, backup python files, and backup the hyperparameter config to a file
    logger = experiment_utils.setup_experiment(config)

    run(config, logger)


if __name__ == "__main__":
    main()
