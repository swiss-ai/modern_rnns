# modelmamba.py

import math
import torch
from torch import nn

# --- Import MambaBlock and LayerNorm from lib.py ---
from lib import MambaBlock, LayerNorm

def mprint(*args, **kwargs):
    print(*args, **kwargs)

class MambaModel(torch.nn.Module):
    """ Mamba-based Language Model using MambaBlock from lib.py. """
    def __init__(self, config):
        super().__init__()
        self.c = config
        self.name = "Mamba"

        # Input Embedding
        self.input_embedding = nn.Embedding(config.num_input_classes, config.h_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        # Mamba Layers (using MambaBlock from lib.py)
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            self.layers.append(
                MambaBlock(
                    h_dim=config.h_dim,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dt_rank=config.dt_rank,
                    layer_idx=i,
                    name=f"{self.name}/Block{i+1}",
                    device=getattr(config, 'device', None),
                    dt_min=getattr(config, 'dt_min', 0.001),
                    dt_max=getattr(config, 'dt_max', 0.1),
                    dt_init=getattr(config, 'dt_init', 'random'),
                    dt_scale=getattr(config, 'dt_scale', 1.0),
                )
            )

        # Final Layer Normalization (using LayerNorm from lib.py)
        self.ln_f = LayerNorm(config.h_dim, name=f"{self.name}/lnf")

        # Output Linear Layer
        self.linear = nn.Linear(config.h_dim, config.output_size, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

        mprint(f"Initialized MambaModel with {config.n_layers} layers.")

    def get_init_state(self, batch_size, device):
        """ Returns the initial state for all Mamba layers. """
        max_seqlen = getattr(self.c, 'max_seq_len', 2048)
        return [
            layer.allocate_inference_cache(batch_size, max_seqlen, device=device)
            for layer in self.layers
        ]

    def forward(self, x, state=None, log=None):
        """ Forward pass through the Mamba model. """
        # 1. Input Embedding
        x = self.input_embedding(x) * math.sqrt(self.c.h_dim)

        # 2. Forward through Mamba Blocks
        new_states = []
        current_layer_input = x
        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None and i < len(state) else None
            current_layer_input, new_layer_state = layer(
                current_layer_input,
                state=layer_state,
                log=log
            )
            new_states.append(new_layer_state)

        # 3. Final LayerNorm
        final_output = self.ln_f(current_layer_input, log=log)

        # 4. Output Linear projection
        logits = self.linear(final_output)

        return logits, new_states
