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
CUDA_VISIBLE_DEVICES=0 torchrun --standalone languini/projects/gpt/main.py tiny --train_batch_size 16 --debug

## Multi GPU:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=server.example.com --master_port=12300 \
    languini/projects/gpt/main.py tiny \
    --train_batch_size 16 \
    --eval_every 1000 \
    --log_grads_every 1000 \
    --tokens_per_second 36930 \
    --decay_steps 19475 \
    --max_train_steps 19475 \
    --gradient_accumulation_steps 2\
"""

import argparse
import os
import torch

import bit_parity_trainer
import bit_parity_dataset
from modelgpt import Model
from modelgpt import DEFAULT_CONFIG as MODEL_DEFAULT_CONFIG

SEQ_LEN = MODEL_DEFAULT_CONFIG["seq_len"]


def run(device):

    train_ds = bit_parity_dataset.BitParityDatasetIterator(
        batch_size=64,
        sequence_length=SEQ_LEN,
        device=device,
    )

    eval_ds = bit_parity_dataset.BitParityDatasetIterator(
        batch_size=64,
        sequence_length=SEQ_LEN,
        device=device,
    )

    ## Setup Model
    config = MODEL_DEFAULT_CONFIG
    config["device"] = device
    config["seq_len"] = SEQ_LEN
    config["vocab_size"] = 2  # number of output classes for the task
    model = Model(config=config)
    model = model.to(device)

    ## Setup Optimiser
    opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-08)

    trainer = bit_parity_trainer.BitParityTrainer(
        model=model,
        train_loader=train_ds,
        eval_loader=eval_ds,
        optimizer=opt,
        device=device,
    )

    print("Begin training ... ")
    trainer.train()
    print("Done!")


def main():
    """Runs a Languini experiment using a GPT model."""

    parser = argparse.ArgumentParser(description="Languini Experiment")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to use for training",
    )
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Languini Experiment")

    # load the config file
    project_path = os.path.dirname(os.path.abspath(__file__))
    print(f"project path: {project_path}")

    run(device)


if __name__ == "__main__":
    main()
