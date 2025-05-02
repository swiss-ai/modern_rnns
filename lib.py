# lib.py (Cleaned Version - May 2, 2025)
# Contains LayerNorm, Mamba class, and MambaBlock.
# Removed __all__ list and fixed potential whitespace issues.
# Kept MLP/Block/QLSTM placeholders for reference if needed by other parts, but simplified.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# This relies on einops being installed: pip install einops
try:
    from einops import rearrange, repeat
except ImportError:
    print("ERROR: 'einops' package not found. Please install it: pip install einops")
    raise # Stop execution if einops is missing, as Mamba depends on it

# --- Optional Mamba Imports (Attempt, but fallback included) ---
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn = None
    mamba_inner_fn = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

# selective_state_update is not used in the current Mamba implementation here
# try:
#     from mamba_ssm.ops.triton.selective_state_update import selective_state_update
# except ImportError:
#     selective_state_update = None


# === Required Components ===

class LayerNorm(nn.Module):
    """ LayerNorm definition """
    def __init__(self, h_dim, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.weight = nn.Parameter(torch.ones(h_dim))
        self.bias = nn.Parameter(torch.zeros(h_dim))

    def forward(self, x, log=None): 
        y = F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)
        return y

class Mamba(nn.Module):
    """ Mamba layer implementation based on mamba_simple.py """
    def __init__(
        self,
        d_model, # h_dim in your config
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None, # Needed for inference caching logic (if used)
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path and mamba_inner_fn is not None and causal_conv1d_fn is not None
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant": nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random": nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else: raise NotImplementedError("Unsupported dt_init value")
        dt_val = torch.exp( torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt_val + torch.log(-torch.expm1(-dt_val))
        with torch.no_grad(): self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D A parameter
        A_vals = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=factory_kwargs.get("device")), "n -> d n", d=self.d_inner).contiguous()
        A_log = torch.log(A_vals)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=factory_kwargs.get("device")))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def forward(self, hidden_states, state=None, log=None):
        """ Mamba forward pass. """
        conv_state, ssm_state = state if state is not None else (None, None)
        batch, seqlen, dim = hidden_states.shape
        current_use_fast_path = self.use_fast_path and state is None

        if current_use_fast_path:
            try:
                xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
                if self.in_proj.bias is not None: xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
                A = -torch.exp(self.A_log.float())
                out = mamba_inner_fn(
                    xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight,
                    self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias,
                    A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
                )
                new_state = (None, None) # Fast path does not return state
            except Exception as e:
                print(f"WARNING: Mamba fast path failed: {e}. Falling back to slow path.")
                current_use_fast_path = False # Fallback to slow path

        if not current_use_fast_path:
            xz = self.in_proj(hidden_states)
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 2, 1) # (B, d_inner, L)

            # --- Convolution ---
            if conv_state is not None:
                if seqlen == 1 and causal_conv1d_update is not None:
                    x_conv_update = x.squeeze(-1)
                    x = causal_conv1d_update(x_conv_update, conv_state, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, self.activation)
                    x = x.unsqueeze(-1)
                    new_conv_state = conv_state # Inplace update
                else: # Handle seqlen > 1 or missing causal_conv1d_update
                    x_padded = torch.cat([conv_state, x], dim=-1)
                    x = self.act(self.conv1d(x_padded)[..., -seqlen:])
                    new_conv_state = x_padded[..., -(self.d_conv - 1):]
            else: # Handle state is None (first pass)
                if causal_conv1d_fn is None: x = self.act(self.conv1d(x)[..., :seqlen])
                else: x = causal_conv1d_fn(x=x, weight=rearrange(self.conv1d.weight, "d 1 w -> d w"), bias=self.conv1d.bias, activation=self.activation)
                # Calculate state for next step
                if seqlen >= self.d_conv: new_conv_state = x[..., -(self.d_conv-1):]
                else: new_conv_state = F.pad(x, (self.d_conv - 1 - seqlen, 0))

            # --- SSM ---
            x = x.permute(0, 2, 1) # Back to (B, L, d_inner)
            x_dbl = self.x_proj(rearrange(x, 'b l d -> (b l) d'))
            dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B_ssm = rearrange(B_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_ssm = rearrange(C_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

            if selective_scan_fn is None:
                raise NotImplementedError("Selective scan function not found/imported.")
            else:
                y, last_ssm_state = selective_scan_fn(
                    u=x.permute(0, 2, 1), delta=dt, A=A, B=B_ssm, C=C_ssm, D=self.D.float(),
                    z=z.permute(0, 2, 1), # Pass z for silu gating
                    delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
                    return_last_state=True, initial_state=ssm_state,
                )
                new_ssm_state = last_ssm_state

            y = y.permute(0, 2, 1) # (B, L, d_inner)
            out = self.out_proj(y) # (B, L, d_model)
            new_state = (new_conv_state, new_ssm_state) # Combine states

        return out, new_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """ Allocate initial state cache. """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv -1, device=device, dtype=conv_dtype)
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=ssm_dtype)
        return conv_state, ssm_state

class MambaBlock(nn.Module):
    """ Mamba Block Wrapper (Uses Mamba and LayerNorm from this file) """
    def __init__(self, h_dim, d_state, d_conv, expand, dt_rank, layer_idx, name, **kwargs):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.layer_idx = layer_idx
        self.ln1 = LayerNorm(h_dim, name=f"{self.name}.ln1") # Using LayerNorm defined above
        self.mamba = Mamba( # Using Mamba defined above
            d_model=h_dim, d_state=d_state, d_conv=d_conv, expand=expand,
            dt_rank=dt_rank, layer_idx=layer_idx, **kwargs
            # Pass device/dtype from kwargs if needed by Mamba's factory_kwargs
            # device=kwargs.get('device'), dtype=kwargs.get('dtype')
        )

    def forward(self, x, state, log=None):
        residual = x
        x_norm = self.ln1(x, log=log)
        mamba_out, new_state = self.mamba(x_norm, state=state, log=log)
        out = residual + mamba_out
        return out, new_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # Pass relevant kwargs if Mamba needs them (e.g., device, dtype)
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# === Optional: Original QLSTM/MLP Components (Simplified Placeholders) ===
# These are NOT needed for Mamba but kept commented out if you want to compare structure later

# def gelu(x):
#     """Gaussian Error Linear Unit (GELU)"""
#     return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# class MLP(torch.nn.Module):
#     """ Original MLP (Placeholder) """
#     def __init__(self, h_dim, mlp_dim, n_layers, name):
#         super().__init__()
#         self.h_dim, self.mlp_dim, self.n_layers, self.name = h_dim, mlp_dim, n_layers, name
#         self.activation_fn = gelu
#         self.c_fc = nn.Linear(h_dim, mlp_dim, bias=True)
#         self.c_proj = nn.Linear(mlp_dim, h_dim, bias=True)
#     def forward(self, x, log=None):
#         return self.c_proj(self.activation_fn(self.c_fc(x)))

# class MultiHeadQuasiLSTMCell(nn.Module):
#      """ Original QuasiLSTM Cell (Placeholder) """
#      def __init__(self, seq_len, h_dim, head_dim, n_heads, name, block_length, max_seq_len=2048):
#          super().__init__()
#          self.name, self.h_dim = name, h_dim
#          self.dummy_linear = nn.Linear(h_dim, h_dim) # Dummy layer
#      def forward(self, f_in, i_in, z_in, o_in, state, log=None):
#          return f_in, state # Passthrough

# class Block(nn.Module):
#     """ Original QLSTM Block (Placeholder) """
#     def __init__(self, seq_len, h_dim, mlp_dim, head_dim, n_heads, n_layers, block_length, name):
#         super().__init__()
#         self.name, self.h_dim, self.seq_len = name, h_dim, seq_len
#         self.ln1 = LayerNorm(h_dim, name=f"{name}.ln1")
#         self.rnn = MultiHeadQuasiLSTMCell(seq_len, h_dim, head_dim, n_heads, name=f"{name}.QLSTM", block_length=block_length)
#         self.ln2 = LayerNorm(h_dim, name=f"{name}.ln2")
#         self.mlp = MLP(h_dim, mlp_dim, n_layers, name=f"{name}.MLP")
#     def forward(self, x, state, log=None):
#         ln_x = self.ln1(x, log=log)
#         rnn_x, new_state = self.rnn(ln_x, ln_x, ln_x, ln_x, state, log=log)
#         x = x + rnn_x
#         mlp_x = self.mlp(self.ln2(x, log=log), log=log)
#         x = x + mlp_x
#         return x, new_state