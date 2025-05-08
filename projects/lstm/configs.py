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

from munch import Munch  # Munch is a dictionary that supports attribute-style access


config_names = [
    "mini",
    # "tiny",
    # 'small',
    # 'medium',
    # 'large',
    # 'XL',
]


def add_exp_name(config):
    """Constructs the name of the log folder used to easily identify the experiment."""
    c = config
    c.exp_name = "{}LSTM{}_{}_sl{}_h{}_ff{}_nH{}_dH{}_nl{}_seed{}{}{}".format(
        "basic",
        "",
        c.dataset,
        c.seq_len,
        c.h_dim,
        c.mlp_dim,
        0,
        0,
        c.n_layers,
        c.seed,
        f"_{c.comment}" if c.comment else "",
        "_debug" if c.debug else "",
    )


## Add experiment configs
def load_config(name=None):

    c = Munch(
        # # data
        # data_root = "data/books",
        relative_log_path="logs",  # Relative path to the log folder within the project folder
        # dataset = "books_16384",
        # vocab_size = 16384,
        debug=False,  # simply adds a "_debug" suffix so logs are easily distinguishable
        # # optimiser
        seed=41,
        # gradient_accumulation_steps = 1,    # number of batches before doing a gradient step
        train_batch_size=1,  # make sure batch sizes are an integer multiple of the number of workers
        eval_batch_size=1,
        test_batch_size=1,
        # seq_len = 512,
        # max_eval_steps = 512,
        # max_train_steps = 500_000,          # total number of training steps
        # decay_steps = 500_000,              # number of steps over which we will decay the learning rate
        # max_lr = 0.0006,                    # starting learning rate
        # min_lr = 0.000006,                  # final learning rate
        # grad_clip_norm = 0.0,               # gradient norm clipping
        # tokens_per_second = 0,              # tokens per second throughput of this config on the hardware run; used for logging over gpuhours
        # # perform certain tasks every N steps
        # eval_every = 1_000,                 # perform a fast evaluation (validation data)
        # test_every = -1,                    # perform a thorough evaluation (test data)
        # log_terminal_every = 100,           # print the current loss to terminal
        # log_metrics_every = 100,            # log accuracy and loss metrics
        # log_grads_every = 1_000,            # log gradients and step sizes
        # log_activations_every = -1,         # log gradients and step sizes
        log_ckpt_every=1_000,  # save model checkpoint to disk
        # logging
        comment="",
        logger_type="wandb",  # can be 'tb', 'wandb' or 'all'
        wandb_project_name="lstm",
    )
    # default model
    if not name or name == "default":
        name = "mini"

    # model
    if name == "mini":
        c.n_layers = 2
        c.h_dim = 4
        c.mlp_dim = 8

        # Dataset config
        c.output_size = 2
        c.num_input_classes = 2

        # Dyck specific
        c.depth = 3
        c.num_parentheses = 4
        c.seq_len = 8

        #MQAR specific
        c.n_keys = 3
        c.n_values = 6
        c.train_num_pairs = "2,3"
        c.eval_num_pairs = "3,6"
        c.unique_keys = True
        c.all_queries_for_input = False

        # Bit parity specific
        c.train_seq_len = "4,8"
        c.eval_seq_len = "8,16"
    else:
        raise ValueError(f"Config name {name} is an invalid name. ")

    return c
