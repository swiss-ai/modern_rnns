import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRU(nn.Module):
    def __init__(self, d_hidden, d_model, r_min=0.0, r_max=1.0, max_phase=6.28):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        # Parameters
        self.nu_log = nn.Parameter(self.nu_init((d_hidden,)))
        self.theta_log = nn.Parameter(self.theta_init((d_hidden,)))
        self.gamma_log = nn.Parameter(self.gamma_log_init())

        # Projections
        self.B_re = nn.Parameter(
            self.matrix_init((d_hidden, d_model), normalization=math.sqrt(2 * d_model))
        )
        self.B_im = nn.Parameter(
            self.matrix_init((d_hidden, d_model), normalization=math.sqrt(2 * d_model))
        )
        self.C_re = nn.Parameter(
            self.matrix_init((d_model, d_hidden), normalization=math.sqrt(d_hidden))
        )
        self.C_im = nn.Parameter(
            self.matrix_init((d_model, d_hidden), normalization=math.sqrt(d_hidden))
        )
        self.D = nn.Parameter(self.matrix_init((d_model,)))

    def matrix_init(self, shape, normalization=1.0):
        return torch.randn(shape) / normalization

    def nu_init(self, shape):
        u = torch.rand(shape)
        return torch.log(
            -0.5 * torch.log(u * (self.r_max**2 - self.r_min**2) + self.r_min**2)
        )

    def theta_init(self, shape):
        u = torch.rand(shape)
        return torch.log(self.max_phase * u)

    def gamma_log_init(self):
        nu, theta = self.nu_log, self.theta_log
        diag_lambda = torch.exp(-torch.exp(nu) + 1j * torch.exp(theta))
        return torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2)).real

    def forward(self, inputs):
        # inputs: [batch, seq_len]
        inputs = inputs.to(torch.cfloat)
        batch_size, seq_len = inputs.shape

        diag_lambda = torch.exp(
            -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
        )
        B_norm = (self.B_re + 1j * self.B_im) * torch.exp(self.gamma_log.unsqueeze(-1))
        C = self.C_re + 1j * self.C_im

        Lambda_elements = diag_lambda.unsqueeze(0).expand(seq_len, -1)  # [T, d_hidden]

        Bu_elements = torch.einsum("hd,td->th", B_norm, inputs)

        h = torch.zeros((self.d_hidden,), dtype=Bu_elements.dtype, device=inputs.device)

        hidden_states = []
        for t in range(seq_len):
            h = Lambda_elements[t] * h + Bu_elements[t]
            hidden_states.append(h.unsqueeze(0))
        hidden_states = torch.cat(hidden_states, dim=0)

        outputs = torch.matmul(hidden_states, C.T).real + inputs * self.D
        return outputs.to(torch.float32)


class SequenceLayer(nn.Module):
    def __init__(self, lru, d_model, dropout=0.0, norm="layer"):
        super().__init__()
        self.lru = lru
        self.d_model = d_model
        self.dropout = dropout
        self.norm_type = norm

        if self.norm_type == "layer":
            self.normalization = nn.LayerNorm(d_model)
        elif self.norm_type == "batch":
            # TODO: running average
            self.normalization = nn.BatchNorm1d(d_model)
        else:
            raise ValueError("Unsupported normalization")

        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs: [batch, seq_len, d_model]
        # TODO: check this
        # if isinstance(self.normalization, nn.BatchNorm1d):
        #     x = self.normalization(inputs.transpose(1, 2)).transpose(1, 2)
        # else:
        x = self.normalization(inputs)

        x = self.lru(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.out1(x) * torch.sigmoid(self.out2(x))
        x = self.drop(x)
        return inputs + x  # skip connection


class StackedEncoderModel(nn.Module):
    def __init__(self, h_dim, d_model, n_layers, dropout=0.0, norm="batch"):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList(
            [
                SequenceLayer(
                    lru=LRU(d_hidden=h_dim, d_model=d_model),
                    d_model=d_model,
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs):
        x = self.encoder(inputs)
        for layer in self.layers:
            x = layer(x)
        return x


class LRUModel(nn.Module):
    def __init__(
        self,
        config,
        dropout=0.0,
        pooling="mean",
        norm="batch",
        multidim=1,
    ):
        super().__init__()
        self.c = c = config
        self.encoder = StackedEncoderModel(
            c.h_dim, c.max_seq_len, c.n_layers, dropout, norm
        )
        self.decoder = nn.Linear(c.max_seq_len, c.output_size * multidim)
        self.pooling = pooling
        self.multidim = multidim

    def get_init_state(self, batch_size, device):
        return None

    def forward(self, inputs, state):
        inputs = inputs.to(torch.float32)
        x = self.encoder(inputs)

        if self.pooling == "mean":
            x = x.mean(dim=1)  # mean across time
        elif self.pooling == "last":
            x = x[:, -1, :]
        elif self.pooling == "none":
            x = x
        else:
            raise ValueError("Unsupported pooling method")

        x = self.decoder(x)

        if self.multidim > 1:
            x = x.view(x.size(0), -1, self.multidim)

        return F.log_softmax(x, dim=-1), state
