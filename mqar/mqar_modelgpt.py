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
import numpy as np

from torch import nn
from lib import LayerNorm
from lib import Block

DEFAULT_CONFIG = {
    "device": "cpu",
    "vocab_size": 2,
    "n_layers": 2,
    "h_dim": 16,
    "mlp_dim": 16,
    "head_dim": 8,
    "n_heads": 4,
    "use_flash": False,
    "seq_len": 32,
    "emb_dim": 32,         
    "value_size": 8,         
    "key_size": 8,         
}

class Config:
    def __init__(self, conf: dict):
        for key, val in conf.items():
            setattr(self, key, val)

class Model(torch.nn.Module):
    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__()
        self.c = c = Config(config)
        self.name = "MQARModel"

        self.key_embedding = nn.Embedding(c.key_size + 1, c.emb_dim)
        self.value_embedding = nn.Embedding(c.value_size + c.key_size + 1, c.emb_dim)
        torch.nn.init.normal_(self.key_embedding.weight, mean=0.0, std=0.02)

        self.key_value_proj = nn.Linear(2 * c.emb_dim, c.h_dim)

        self.position_embedding = nn.Embedding(c.seq_len, c.h_dim)
        torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            Block(
                h_dim=c.h_dim,
                mlp_dim=c.mlp_dim,
                head_dim=c.head_dim,
                n_heads=c.n_heads,
                n_layers=c.n_layers,
                name=f"{self.name}/Block{i+1}",
                use_flash=c.use_flash,
            ) for i in range(c.n_layers)
        ])

        self.ln_f = LayerNorm(c.h_dim, name=f"{self.name}/lnf")

        self.linear = nn.Linear(c.h_dim, c.vocab_size, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def get_init_state(self, batch_size, device):
        return None

    def forward(self, x, state=None, log=None):
        #   x[0]: key indices [seq_len]
        #   x[1]: one-hot value vectors [seq_len, value_size]

        c = self.c
        keys = x[:, 0, :]  
        values = x[:, 1, :]
        seq_len = keys.shape[1]

        key_emb = self.key_embedding(keys)  # [seq_len, emb_dim]
        value_emb = self.value_embedding(values)
        combined = torch.cat([key_emb, value_emb], dim=-1)  # [seq_len, emb_dim + value_size]

        x = self.key_value_proj(combined)  # [seq_len, h_dim]
        x = x.unsqueeze(0)  # [1, seq_len, h_dim]

        pos_id = torch.arange(seq_len, device=c.device).unsqueeze(0)  # [1, seq_len]
        pos = self.position_embedding(pos_id)  # [1, seq_len, h_dim]
        x = x + pos
        x = x.squeeze(0)

        for layer in self.layers:
            x = layer(x, log=log)

        x = self.ln_f(x, log=log)
        logits = self.linear(x)  # [1, seq_len, vocab_size]
        return logits, state

if __name__ == "__main__":
    c = Config(DEFAULT_CONFIG)
    c.h_dim = 128
    c.emb_dim = 64
    c.mlp_dim = 256
    c.head_dim = 16
    c.n_heads = 8
    c.n_layers = 2
    c.value_size = 16
    c.batch_size = 1
    c.seq_len = 10
    c.vocab_size = 32
    c.use_flash = False
    c.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(config=c).to(c.device)

    keys = torch.randint(0, c.vocab_size, (c.seq_len,), device=c.device)
    values = torch.eye(c.value_size)[torch.randint(0, c.value_size, (c.seq_len,))].to(c.device)
    x = [keys, values]

    state = model.get_init_state(batch_size=c.batch_size, device=c.device)
    logits, state = model(x, state)
    print(f"{logits.shape=}")