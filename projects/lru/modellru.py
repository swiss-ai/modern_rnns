import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from projects.lru.lib import StackedEncoderModel, LRU

# Modified LRUModel to output logits per time step when pooling="none"
class LRUModel(nn.Module):
    def __init__(
        self,
        config,
        dropout=0.0,
        pooling="none", # Keep pooling option
        norm="batch",
        multidim=1,
        input_features=1 # Add input_features, default 1 for bit parity task
    ):
        super().__init__()
        self.c = c = config
        self.input_features = input_features
        self.pooling = pooling
        self.multidim = multidim

        self.encoder = StackedEncoderModel(
            input_features=self.input_features, 
            h_dim=c.h_dim,
            d_model=c.d_model,
            n_layers=c.n_layers,
            dropout=dropout,
            norm=norm
        )

        self.decoder = nn.Linear(c.d_model, c.output_size * multidim)


    def get_init_state(self, batch_size, device):
        return None

    def forward(self, inputs, state=None):
        x = self.encoder(inputs.to(torch.float32)) # Cast input to float32 before encoder

        # Apply decoder based on pooling mode
        if self.pooling == "mean":
            x_pooled = x.mean(dim=1)  # [batch, d_model]
            decoded_output = self.decoder(x_pooled) # [batch, output_size * multidim]

        elif self.pooling == "last":
            x_pooled = x[:, -1, :] # [batch, d_model]
            decoded_output = self.decoder(x_pooled) # [batch, output_size * multidim]

        elif self.pooling == "none":
            decoded_output = self.decoder(x) # [batch, seq_len, output_size * multidim]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")


        # Reshape for multidim if needed
        if self.multidim > 1:
            final_output_shape = decoded_output.shape[:-1] + (self.c.output_size, self.multidim)
            reshaped_output = decoded_output.view(final_output_shape)
        else:
            reshaped_output = decoded_output

        logits = F.log_softmax(reshaped_output, dim=-1)

        return logits, state 