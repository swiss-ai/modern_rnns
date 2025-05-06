import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep LRU class as is (from the previous corrected version)
class LRU(nn.Module):
    def __init__(self, d_hidden, d_model, r_min=0.9, r_max=0.99, max_phase=6.28):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model # This is the feature dimension of the input/output
        self.r_min = r_min  # Initialize here
        self.r_max = r_max
        self.max_phase = math.pi / 50

        if r_max > 1.0:
             pass

        # Parameters
        self.nu_log = nn.Parameter(self.nu_init((d_hidden,)))
        self.theta_log = nn.Parameter(self.theta_init((d_hidden,)))
        self.gamma_log = nn.Parameter(self.gamma_log_init(self.nu_log, self.theta_log))

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
        u = torch.rand(shape) * (1 - 1e-5) + 1e-5 # Avoid log(0) or log(very small)
        magnitude_sq = u * (self.r_max**2 - self.r_min**2) + self.r_min**2

        nu = torch.log(-0.5 * torch.log(magnitude_sq + 1e-5))
        return nu

    def theta_init(self, shape):
        u = torch.rand(shape) * (1 - 1e-5) + 1e-5
        return torch.log(self.max_phase * u)  # small phase

    def gamma_log_init(self, nu, theta):
        gamma_log = 0.5 * torch.log(1 - torch.exp(-2 * torch.exp(nu)) + 1e-5)
        return gamma_log # Should be real

    def forward(self, inputs):
        batch_size, seq_len, d_model_in = inputs.shape

        # Ensure d_model matches the input feature dimension
        if d_model_in != self.d_model:
             raise ValueError(f"Input feature dimension ({d_model_in}) must match LRU's d_model ({self.d_model})")

        # Ensure inputs is complex for state operations
        inputs_c = inputs.to(torch.cfloat)

        nu = torch.exp(self.nu_log)    # Ensure nu is positive real
        theta = torch.exp(self.theta_log) # Ensure theta is positive real
        diag_lambda = torch.exp(-nu + 1j * theta) # [d_hidden]

        gamma_log = self.gamma_log_init(self.nu_log, self.theta_log) # [d_hidden]
        
        B_norm = (self.B_re + 1j * self.B_im) * torch.exp(gamma_log.unsqueeze(-1)) # [d_hidden, d_model]

        Bu_elements = torch.einsum("hd,btd->bth", B_norm, inputs_c) # [batch_size, seq_len, d_hidden]

        h = torch.zeros((batch_size, self.d_hidden,), dtype=inputs_c.dtype, device=inputs_c.device) # [batch_size, d_hidden]

        hidden_states = [] # To collect h for all time steps

        for t in range(seq_len):
            h = diag_lambda * h + Bu_elements[:, t, :]
            hidden_states.append(h.unsqueeze(1)) # Add time dimension back: [batch_size, 1, d_hidden]

        # Concatenate hidden states along the time dimension
        hidden_states = torch.cat(hidden_states, dim=1) # [batch_size, seq_len, d_hidden]

        C_complex = self.C_re + 1j * self.C_im # [d_model, d_hidden]
        outputs_rnn = torch.matmul(hidden_states, C_complex.T) # [batch_size, seq_len, d_model]

        outputs = outputs_rnn.real + inputs_c.real * self.D # Using real parts for the final float output

        # Final output should be float32
        return outputs.to(torch.float32)


# Keep SequenceLayer class as is (from the previous corrected version)
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
            self.normalization = nn.BatchNorm1d(d_model)
        else:
            raise ValueError("Unsupported normalization")

        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs: [batch, seq_len, d_model]

        x = inputs # Start with inputs
        if self.norm_type == "layer":
             x = self.normalization(x) # Input/Output: [batch, seq_len, d_model]
        elif self.norm_type == "batch":
             x = x.transpose(1, 2) # [batch, d_model, seq_len]
             x = self.normalization(x) 
             x = x.transpose(1, 2) 
        else:
             raise ValueError("Unsupported normalization")

        # x is now [batch, seq_len, d_model]
        x = self.lru(x) # LRU takes [batch, seq_len, d_model] and returns [batch, seq_len, d_model]

        x = F.gelu(x) # [batch, seq_len, d_model]
        x = self.drop(x) # [batch, seq_len, d_model]

        x = self.out1(x) * torch.sigmoid(self.out2(x)) # [batch, seq_len, d_model] * [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        x = self.drop(x) # [batch, seq_len, d_model]

        # Skip connection
        return inputs + x  # [batch, seq_len, d_model] + [batch, seq_len, d_model]


class StackedEncoderModel(nn.Module):
    def __init__(self, input_features, h_dim, d_model, n_layers, dropout=0.0, norm="batch"):
        super().__init__()
        self.input_features = input_features
        self.d_model = d_model

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_features, d_model)

        # The rest of the layers operate within the d_model dimension
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
        if inputs.ndim == 2 and self.input_features == 1:
            inputs = inputs.unsqueeze(-1) # [batch, seq_len, 1]

        if inputs.shape[-1] != self.input_features:
             raise ValueError(f"Input last dimension ({inputs.shape[-1]}) must match StackedEncoderModel's input_features ({self.input_features})")


        # Project input features to d_model
        x = self.input_projection(inputs) # [batch, seq_len, input_features] -> [batch, seq_len, d_model]

        # Pass through LRU layers
        for layer in self.layers:
            x = layer(x) # [batch, seq_len, d_model] -> [batch, seq_len, d_model]

        return x # [batch, seq_len, d_model]