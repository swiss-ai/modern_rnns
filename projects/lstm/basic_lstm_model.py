import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c = c = config
        self.name = "BasicLSTM"

        self.embedding = nn.Embedding(c.num_input_classes, c.h_dim)
        self.lstm = nn.LSTM(
            input_size=c.h_dim,
            hidden_size=c.h_dim,
            num_layers=c.n_layers,
            batch_first=True,
        )

        self.output = nn.Linear(c.h_dim, c.output_size)

    def get_init_state(self, batch_size, device):
        # (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.c.n_layers, batch_size, self.c.h_dim, device=device)
        c0 = torch.zeros(self.c.n_layers, batch_size, self.c.h_dim, device=device)
        return (h0, c0)

    def forward(self, x, state, log=None):
        # x: [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.c.h_dim)  # [B, T, H]

        # Run through LSTM
        x, new_state = self.lstm(x, state)  # [B, T, H], state = (h_n, c_n)

        # Project to output
        logits = self.output(x)  # [B, T, output_size]

        return logits, new_state
