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
from torch import nn

# from munch import Munch
# from languini.train_lib.train_utils import check_config
from common_lib.debug_utils import check
# from languini.common_lib.debug_utils import log_stats_and_dist

from projects.linearTransformer.lib import LayerNorm
from projects.linearTransformer.lib import Block


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # check_config(config, DEFAULT_CONFIG)
        self.c = c = config
        self.name = "LinearTransformer"

        self.input_embedding = nn.Embedding(c.num_input_classes, c.h_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        self.position_embedding = nn.Embedding(c.max_seq_len, c.h_dim)
        torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([])
        for i in range(c.n_layers):
            self.layers.append(
                Block(
                    h_dim=c.h_dim,
                    mlp_dim=c.mlp_dim,
                    head_dim=c.head_dim,
                    n_heads=c.n_heads,
                    n_layers=c.n_layers,
                    name=f"{self.name}/Block{i + 1}",
                    use_flash=c.use_flash,
                )
            )

        self.ln_f = LayerNorm(c.h_dim, name=f"{self.name}/lnf")

        self.linear = nn.Linear(c.h_dim, c.output_size, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def get_init_state(self, batch_size, device):
        return None

    def forward(self, x, state, log=None):
        # x: [batch_size, seq_length]
        bsz, seqlen = x.shape
        c = self.c
        # print("input",x.shape)

        # embedd input tokens
        x = self.input_embedding(x) * math.sqrt(c.h_dim)
        # print("embeddings",x.shape)

        check(x, (bsz, seqlen, c.h_dim))

        # add position embedding
        pos_id = torch.arange(0, seqlen, dtype=torch.int64, device=c.device).unsqueeze(
            0
        )
        # check(pos_id, (1, seqlen))
        pos = self.position_embedding(pos_id)
        check(pos, (1, seqlen, c.h_dim))
        x = x + pos

        # forward
        for layer in self.layers:
            x = layer(x, log=log)
            # check(x, (bsz, seqlen, c.h_dim))

        # print("after layers", x.shape)

        # project to vocab
        x = self.ln_f(x, log=log)
        # print("before mean x", x.shape)

        logits = self.linear(x)
        # print("logits", x.shape)
        # check(x, (bsz, c.vocab_size))

        return logits, state
