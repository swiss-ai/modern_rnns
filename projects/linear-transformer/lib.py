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


class LinearAttention(nn.Module):
    def __init__(self, seq_len, h_dim, head_dim, n_heads, block_length, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        assert h_dim == n_heads * head_dim, "h_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(h_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)
        self.out_proj = nn.Linear(h_dim, h_dim)

        self.feature_map = lambda x: F.elu(x) + 1  # Feature map for kernelized attention

    def forward(self, f_in, i_in, z_in, o_in, state=None, log=None):
        # f_in: Input tensor (batch_size, seq_len, h_dim)
        batch_size, seq_len, _ = f_in.size()
        num_heads = self.n_heads
        dim_per_head = self.head_dim

        query = self.q_proj(f_in).view(batch_size, seq_len, num_heads, dim_per_head).transpose(1, 2)  # (B, H, T, D)
        key = self.k_proj(i_in).view(batch_size, seq_len, num_heads, dim_per_head).transpose(1, 2)
        value = self.v_proj(z_in).view(batch_size, seq_len, num_heads, dim_per_head).transpose(1, 2)

        query_features = self.feature_map(query)
        key_features = self.feature_map(key)

        # Precompute key-value summary
        key_value_summary = torch.einsum("bhtd,bhte->bhde", key_features, value)  # (B, H, D, D)
        normalization = 1 / (torch.einsum("bhtd,bhd->bht", query_features, key_features.sum(dim=2)) + 1e-6)  # (B, H, T)

        output = torch.einsum("bhtd,bhde->bhte", query_features, key_value_summary)  # (B, H, T, D)
        output = output * normalization.unsqueeze(-1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # (B, T, H*D)

        return self.out_proj(output), state


class Block(nn.Module):
    def __init__(self, seq_len, h_dim, mlp_dim, head_dim, n_heads, n_layers, block_length, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.seq_len = seq_len

        self.ln1 = LayerNorm(h_dim, name=f"{self.name}.ln1")

        self.rnn = LinearAttention(seq_len=seq_len, h_dim=h_dim, head_dim=head_dim, n_heads=n_heads,
                                              block_length=block_length,
                                              name=f"{self.name}.LinearAttention")

        self.ln2 = LayerNorm(h_dim, name=f"{self.name}.ln2")
        self.mlp = MLP(h_dim=h_dim, mlp_dim=mlp_dim, n_layers=n_layers, name=f"{self.name}.MLP")

    def forward(self, x, state, log=None):
        bsz, _, _ = x.shape
        check(x, (bsz, self.seq_len, self.h_dim))

        ln_x = self.ln1(x, log=log)
        rnn_x, new_state = self.rnn(f_in=ln_x, i_in=ln_x, z_in=ln_x, o_in=ln_x, state=state, log=log)
        x = x + rnn_x

        mlp_x = self.mlp(self.ln2(x, log=log), log=log)
        x = x + mlp_x

        return x, new_state