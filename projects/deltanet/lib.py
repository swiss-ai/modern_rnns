# GPT-2 component implementations based on the official TensorFlow code https://github.com/openai/gpt-2/blob/master/src/model.py

import math
import torch
import torch.nn.functional as F

from torch import nn
from common_lib.debug_utils import check
# from languini.common_lib.debug_utils import log_stats_and_dist


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU)
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def phi(x):
    return F.elu(x)+1


class LayerNorm(nn.Module):
    def __init__(self, h_dim, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.weight = nn.Parameter(torch.ones(h_dim))
        self.bias = nn.Parameter(torch.zeros(h_dim))

    def forward(self, x, log=None):
        if torch.isnan(x).any():
            print("bad x")
        x = torch.clamp(x, -1e4, 1e4)

        bsz, seqlen, _ = x.shape
        check(x, (bsz, seqlen, self.h_dim))

        # log_stats_and_dist(x, f"{self.name}.preLN", log)
        y = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        # log_stats_and_dist(x, f"{self.name}.postLN", log)
        if torch.isnan(y).any():
            print("bad ln")
        return y


class DeltaRule(nn.Module):
    def __init__(self, h_dim, head_dim, n_heads, name, use_flash, max_seq_len=4096):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.use_flash = use_flash

        self.linear_Q = nn.Linear(h_dim, head_dim * n_heads, bias=True)
        torch.nn.init.normal_(self.linear_Q.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_Q.bias)

        self.linear_K = nn.Linear(h_dim, head_dim * n_heads, bias=True)
        torch.nn.init.normal_(self.linear_K.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_K.bias)

        self.linear_V = nn.Linear(h_dim, head_dim * n_heads, bias=True)
        torch.nn.init.normal_(self.linear_V.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_V.bias)

        self.linear_O = nn.Linear(head_dim * n_heads, h_dim, bias=True)
        torch.nn.init.normal_(self.linear_O.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_O.bias)

        self.linear_beta = nn.Linear(h_dim, n_heads, bias=True)
        torch.nn.init.normal_(self.linear_beta.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear_beta.bias)

        self.sigma = nn.Sigmoid()

        # construct and store the following objects during initialisation for a faster forward pass
        self.dot_scale = float(head_dim) ** -0.5
        self.register_buffer("mask",
                            torch.tril(torch.ones(max_seq_len, max_seq_len))
                            .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, query, key, value, beta, log=None):
        bsz, seqlen, _ = query.shape
        check(query, (bsz, seqlen, self.h_dim))
        check(key, (bsz, seqlen, self.h_dim))
        check(value, (bsz, seqlen, self.h_dim))

        q = self.linear_Q(query)
        k = self.linear_K(key)
        v = self.linear_V(value)
        b = self.linear_beta(beta)


        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        check(q, (bsz, self.n_heads, seqlen, self.head_dim))

        b = b.view(bsz, seqlen, self.n_heads).transpose(1, 2)
        b = self.sigma(b) # [b, h, s]
        check(b, (bsz, self.n_heads, seqlen))

        phi_q = phi(q)  # [b, h, s, d]
        phi_k = phi(k)  # [b, h, s, d]


        prev_W = torch.zeros(bsz, self.n_heads, self.head_dim, self.head_dim)
        out = list()

        for i in range(seqlen):
            v_i_bar = torch.einsum("bhdd, bhd -> bhd", prev_W, phi_k[:, :, i])  # [b, h, d]
            check(v_i_bar, (bsz, self.n_heads, self.head_dim))

            # weighed average of v and v_bar
            # v_new = bv + (1-b)v_bar = b(v-v_bar) + v_bar
            beta_delta_v = torch.einsum("bh, bhd -> bhd", b[:, :, i], (v[:, :, i] - v_i_bar))
            check(beta_delta_v, (bsz, self.n_heads, self.head_dim))
            v_new_i = beta_delta_v + v_i_bar  # [b,h,d]
            check(v_new_i, (bsz, self.n_heads, self.head_dim))

            # update W
            update = torch.einsum("bhd, bhe -> bhde", beta_delta_v, phi_k[:, :, i])
            # if torch.isnan(phi_k[:, :, i]).any():
            #     print(phi_k[:, :, i].shape)
            check(update, (bsz, self.n_heads, self.head_dim, self.head_dim))
            curr_W = prev_W + update     # [b, h, d, d]
            check(curr_W, (bsz, self.n_heads, self.head_dim, self.head_dim))

            y_i = torch.einsum("bhdd, bhd -> bhd", curr_W, phi_q[:, :, i])  # [b, h, d]
            check(y_i, (bsz, self.n_heads, self.head_dim))
            out.append(y_i)

            prev_W = curr_W


        # Merge heads and project
        # out is  [[b, h, d] * s]
        out = torch.stack(out, dim=1)
        check(out, (bsz, seqlen, self.n_heads, self.head_dim))
        out = out.reshape(bsz, seqlen, self.n_heads * self.head_dim)
        y = self.linear_O(out)
        return y

    def __repr__(self):
        return f"CausalSelfAttention(h_dim={self.h_dim}, head_dim={self.head_dim}, n_heads={self.n_heads}, name={self.name})"


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
        # log_stats_and_dist(x, f"{self.name}.pre_act", log)

        x = self.activation_fn(x)
        # log_stats_and_dist(x, f"{self.name}.post_act", log)

        x = self.c_proj(x)
        check(x, (bsz, seq_len, self.h_dim))

        return x

    def __repr__(self):
        return f"MLP(h_dim={self.h_dim}, mlp_dim={self.mlp_dim}, activation_fn={self.activation_fn}, name={self.name})"


class Block(nn.Module):
    def __init__(self, h_dim, mlp_dim, head_dim, n_heads, n_layers, name, use_flash, max_seq_len=4096):
        super().__init__()
        self.name = name
        self.use_flash = use_flash
        self.h_dim = h_dim
        self.ln1 = LayerNorm(h_dim, name=f"{self.name}.ln1")
        self.attn = DeltaRule(h_dim=h_dim, head_dim=head_dim, n_heads=n_heads,
                                        name=f"{self.name}.CausalAttn",
                                        use_flash=self.use_flash,
                                        max_seq_len=max_seq_len)
        self.ln2 = LayerNorm(h_dim, name=f"{self.name}.ln2")
        self.mlp = MLP(h_dim=h_dim, mlp_dim=mlp_dim, n_layers=n_layers, name=f"{self.name}.MLP")

    def forward(self, x, log=None):
        bsz, seqlen, _ = x.shape
        # check(x, (bsz, seqlen, self.h_dim))

        ln_x = self.ln1(x, log=log)
        attn_x = self.attn(query=ln_x, key=ln_x, value=ln_x, beta=ln_x, log=log)
        # log_stats_and_dist(attn_x, f"{self.name}.attn_delta", log)
        x = x + attn_x

        mlp_x = self.mlp(self.ln2(x, log=log), log=log)

        x = x + mlp_x

        return x