import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_layers = config.n_layers

        self.embedding = nn.Embedding(
            num_embeddings=config.num_input_classes,
            embedding_dim=config.h_dim
        )

        self.lstm = nn.LSTM(
            input_size=config.h_dim,
            hidden_size=config.h_dim,
            num_layers=config.n_layers,
            batch_first=True
        )

        self.fc = nn.Linear(config.h_dim, config.output_size)

    def get_init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.config.h_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.config.h_dim, device=device)
        return [(h0[i], c0[i]) for i in range(self.num_layers)]

    def forward(self, x, state=None):
        x = self.embedding(x)

        batch_size = x.size(0)

        if state is None:
            state = self.get_init_state(batch_size, x.device)

        h0 = torch.stack([s[0] for s in state], dim=0)
        c0 = torch.stack([s[1] for s in state], dim=0)

        out, (hn, cn) = self.lstm(x, (h0, c0)) 
        out = self.fc(out)

        new_states = [(hn[i], cn[i]) for i in range(self.num_layers)]

        return out, new_states
