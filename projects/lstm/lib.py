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
import torch.nn.functional as F

from torch import nn
from common_lib.debug_utils import check


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU)
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    def __init__(self, h_dim, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.weight = nn.Parameter(torch.ones(h_dim))
        self.bias = nn.Parameter(torch.zeros(h_dim))

    def forward(self, x, log=None):
        bsz, seqlen, _ = x.shape
        check(x, (bsz, seqlen, self.h_dim))

        y = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        return y


class MultiHeadLSTMCell(nn.Module):
    """Implements an LSTM cell with multiple heads."""

    def __init__(self, h_dim, head_dim, n_heads, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.head_dim = head_dim
        self.n_heads = n_heads

        # linear projections for x (done in parallel)
        self.linear_Fx = nn.Linear(h_dim, head_dim * n_heads, bias=True)
        torch.nn.init.normal_(self.linear_Fx.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Fx.bias)

        self.linear_Ix = nn.Linear(h_dim, head_dim * n_heads, bias=True)
        torch.nn.init.normal_(self.linear_Ix.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Ix.bias)

        self.linear_Zx = nn.Linear(h_dim, head_dim * n_heads, bias=True)
        torch.nn.init.normal_(self.linear_Zx.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Zx.bias)

        self.linear_Ox = nn.Linear(h_dim, head_dim * n_heads, bias=True)
        torch.nn.init.normal_(self.linear_Ox.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Ox.bias)

        # linear projections of the state (done in sequence)
        self.linear_Fh = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=True)
        torch.nn.init.normal_(self.linear_Fh.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Fh.bias)

        self.linear_Ih = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=True)
        torch.nn.init.normal_(self.linear_Ih.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Ih.bias)

        self.linear_Zh = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=True)
        torch.nn.init.normal_(self.linear_Zh.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Zh.bias)

        self.linear_Oh = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=True)
        torch.nn.init.normal_(self.linear_Oh.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Oh.bias)

        # additional linear layer to project all heads back to h_dim in case of multiple heads (as in attention)
        if self.n_heads > 1:
            self.linear_post_head = nn.Linear(head_dim * n_heads, h_dim, bias=True)
            torch.nn.init.normal_(self.linear_post_head.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.linear_post_head.bias)

    def forward(self, f_in, i_in, z_in, o_in, state, log=None):
        bsz, seq_len, _ = f_in.shape
        c, h = state
        check(c, (bsz, self.n_heads, self.head_dim))
        check(h, (bsz, self.n_heads, self.head_dim))

        check(f_in, (bsz, seq_len, self.h_dim))
        check(i_in, (bsz,seq_len, self.h_dim))
        check(z_in, (bsz, seq_len, self.h_dim))
        check(o_in, (bsz, seq_len, self.h_dim))

        # compute the gates given x in parallel
        fx = self.linear_Fx(f_in)  # forget gate
        ix = self.linear_Ix(i_in)  # input gate
        zx = self.linear_Zx(z_in)  # cell update
        ox = self.linear_Ox(o_in)  # output gate

        fx = fx.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        ix = ix.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        zx = zx.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        ox = ox.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        check(fx, (bsz, self.n_heads, seq_len, self.head_dim))

        # iterate over the sequence
        outputs = []
        for idx in range(seq_len):
            check(h, (bsz, self.n_heads, self.head_dim))
            h = h.view(bsz, self.n_heads * self.head_dim)

            # gate contribution of the current state
            fh = self.linear_Fh(h)  # forget gate
            ih = self.linear_Ih(h)  # input gate
            zh = self.linear_Zh(h)  # cell update
            oh = self.linear_Oh(h)  # output gate

            fh = fh.view(bsz, self.n_heads, self.head_dim)
            ih = ih.view(bsz, self.n_heads, self.head_dim)
            zh = zh.view(bsz, self.n_heads, self.head_dim)
            oh = oh.view(bsz, self.n_heads, self.head_dim)
            check(fh, (bsz, self.n_heads, self.head_dim))

            f = torch.sigmoid(fx[:, :, idx, :] + fh)
            i = torch.sigmoid(ix[:, :, idx, :] + ih)
            z = torch.tanh(zx[:, :, idx, :] + zh)
            o = torch.sigmoid(ox[:, :, idx, :] + oh)
            check(f, (bsz, self.n_heads, self.head_dim))

            c = f * c + i * z
            h = o * torch.tanh(c)
            check(c, (bsz, self.n_heads, self.head_dim))
            check(h, (bsz, self.n_heads, self.head_dim))
            outputs.append(h)

        state = c, h

        # project all outputs
        outputs = torch.stack(outputs, dim=1)
        check(outputs, (bsz, seq_len, self.n_heads, self.head_dim))
        y = outputs.reshape(bsz, seq_len, self.n_heads * self.head_dim)
        if self.n_heads > 1:
            y = self.linear_post_head(y)
        check(y, (bsz, seq_len, self.h_dim))

        return y, state

    def __repr__(self):
        return f"MultiHeadLSTMCell(h_dim={self.h_dim}, head_dim={self.head_dim}, n_heads={self.n_heads}, name={self.name})"


class MLP(torch.nn.Module):
    def __init__(self, h_dim, mlp_dim, n_layers, name):
        super().__init__()
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers
        self.name = name
        self.activation_fn = gelu

        self.c_fc = nn.Linear(h_dim, mlp_dim, bias=True)
        torch.nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.c_fc.bias)

        self.c_proj = nn.Linear(mlp_dim, h_dim, bias=True)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        torch.nn.init.zeros_(self.c_proj.bias)

    def forward(self, x, log=None):
        bsz, seq_len, _ = x.shape

        check(x, (bsz, seq_len, self.h_dim))

        x = self.c_fc(x)
        check(x, (bsz, seq_len, self.mlp_dim))

        x = self.activation_fn(x)

        x = self.c_proj(x)
        check(x, (bsz, seq_len, self.h_dim))

        return x

    def __repr__(self):
        return f"MLP(h_dim={self.h_dim}, mlp_dim={self.mlp_dim}, activation_fn={self.activation_fn}, name={self.name})"


class Block(nn.Module):
    def __init__(self, h_dim, mlp_dim, head_dim, n_heads, n_layers, block_length, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim

        self.ln1 = LayerNorm(h_dim, name=f"{self.name}.ln1")

        self.rnn = MultiHeadLSTMCell(h_dim=h_dim, head_dim=head_dim, n_heads=n_heads,
                                  name=f"{self.name}.MultiHeadLSTM")

        self.ln2 = LayerNorm(h_dim, name=f"{self.name}.ln2")
        self.mlp = MLP(h_dim=h_dim, mlp_dim=mlp_dim, n_layers=n_layers, name=f"{self.name}.MLP")

    def forward(self, x, state, log=None):
        bsz, seq_len, _ = x.shape
        check(x, (bsz, seq_len, self.h_dim))

        ln_x = self.ln1(x, log=log)
        rnn_x, new_state = self.rnn(f_in=ln_x, i_in=ln_x, z_in=ln_x, o_in=ln_x, state=state, log=log)
        x = x + rnn_x

        mlp_x = self.mlp(self.ln2(x, log=log), log=log)
        x = x + mlp_x

        return x, new_state