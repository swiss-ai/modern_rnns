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

import math
import torch
# import numpy as np

from torch import nn
from torch.nn import LayerNorm

from projects.lstm.lib import LayerNorm
from projects.lstm.lib import Block

# c.h_dim = 128
#     c.mlp_dim = 256
#     c.head_dim = 16
#     c.n_heads = 8
#     c.n_layers = 2
#
#     c.batch_size = 8
#     c.block_length = 32
#     c.seq_len = 512
#     c.vocab_size = 666

DEFAULT_CONFIG = {
    "device": None,
    "vocab_size": 2,#
    "n_layers": 1,
    "h_dim": 16,
    "mlp_dim": 8,
    "head_dim": 8, #change for q lstm
    "n_heads": 8, #change for q lstm
    "non_quasi": False,
    "block_length": 8,
    "seq_len": 8
}

class Config():
    def __init__(self, conf: dict):
        for key, val in conf.items():
            setattr(self, key, val)
class Model(torch.nn.Module):
    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__()
        self.c = c = Config(config)
        self.name = "LSTM"

        self.input_embedding = nn.Embedding(c.vocab_size, c.h_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        # self.position_embedding = nn.Embedding(c.seq_len, c.h_dim)
        # torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([])
        for i in range(c.n_layers):
            self.layers.append(Block(seq_len=c.seq_len,
                                     h_dim=c.h_dim,
                                     mlp_dim=c.mlp_dim,
                                     head_dim=c.head_dim,
                                     n_heads=c.n_heads,
                                     n_layers=c.n_layers,
                                     non_quasi=c.non_quasi,
                                     block_length=c.block_length,
                                     name=f"{self.name}/Block{i + 1}"))

        self.ln_f = LayerNorm(c.h_dim, name=f"{self.name}/lnf")

        self.linear = nn.Linear(c.h_dim, c.vocab_size, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def get_init_state(self, batch_size, device):
        return [
            (torch.zeros((batch_size, self.c.n_heads, self.c.head_dim), device=device),
             torch.zeros((batch_size, self.c.n_heads, self.c.head_dim), device=device))
            for _ in range(self.c.n_layers)
        ]

    def forward(self, x, state, log=None):
        # x: [batch_size, seq_length]


        # embedd input tokens
        x = self.input_embedding(x) * math.sqrt(self.c.h_dim)


        # forward
        new_states = []
        for idx, layer in enumerate(self.layers):

            x, new_state = layer(x, state[idx], log=log)


            new_states.append(new_state)

        # project to vocab
        x = self.ln_f(x, log=log)
        logits = self.linear(x)

        return logits, new_states


