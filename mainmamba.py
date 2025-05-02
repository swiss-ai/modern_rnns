# mainmamba.py  (clean, full functionality, Dyck + MQAR fixed)

import argparse
import os
import sys
import time
import json

import torch
from munch import Munch  # pip install munch

from configs import load_config, add_exp_name, config_names
from modelmamba import MambaModel
from datasets.bit_parity_dataset import BitParityDatasetIterator
from datasets.dyck_dataset       import DyckDatasetIterator
from datasets.mqar_dataset       import MQARDatasetIterator


def mprint(*args, **kwargs):
    print(*args, **kwargs)


def create_parser_based_on_config(config):
    parser = argparse.ArgumentParser(description="Mamba Model Training Script")
    # core arguments
    parser.add_argument("--dataset",        type=str,   required=True,
                        choices=["bit_parity", "dyck", "mqar"],
                        help="Dataset to use.")
    parser.add_argument("--device",         type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu","cuda"],
                        help="Device to run on (cpu or cuda).")
    parser.add_argument("--train_batch_size", type=int, help="Training batch size.")
    parser.add_argument("--eval_batch_size",  type=int, help="Evaluation batch size.")
    parser.add_argument("--max_train_steps",  type=int, help="Maximum number of training steps.")
    parser.add_argument("--max_seq_len",      type=int, help="Max padding length (bit_parity/dyck).")
    # mamba model overrides
    parser.add_argument("--n_layers",     type=int,   help="Number of Mamba layers.")
    parser.add_argument("--h_dim",        type=int,   help="Model hidden dimension.")
    parser.add_argument("--d_state",      type=int,   help="Mamba state dimension.")
    parser.add_argument("--d_conv",       type=int,   help="Mamba convolution kernel size.")
    parser.add_argument("--expand",       type=int,   help="Mamba expansion factor.")
    parser.add_argument("--lr",           type=float, help="Learning rate.")
    # logging & checkpointing
    parser.add_argument("--eval_every",         type=int, help="Evaluate every N steps.")
    parser.add_argument("--log_terminal_every", type=int, help="Log loss every N steps.")
    parser.add_argument("--log_ckpt_every",     type=int, help="Save checkpoint every N steps.")
    # experiment meta
    parser.add_argument("--debug",        action="store_true", default=None,
                        help="Enable debug mode (suffix _debug).")
    parser.add_argument("--comment",      type=str, help="Custom comment for exp name.")
    # dataset-specific
    parser.add_argument("--train_seq_len",   type=str,
                        help="Min,Max seq length for bit_parity/dyck training.")
    parser.add_argument("--eval_seq_len",    type=str,
                        help="Min,Max seq length for bit_parity/dyck eval.")
    parser.add_argument("--depth",           type=int,
                        help="Max nesting depth for Dyck.")
    parser.add_argument("--num_parentheses", type=int,
                        help="Number of parenthesis types for Dyck.")
    parser.add_argument("--train_num_pairs", type=str,
                        help="Min,Max KV pairs for MQAR training.")
    parser.add_argument("--eval_num_pairs",  type=str,
                        help="Min,Max KV pairs for MQAR eval.")
    parser.add_argument("--n_keys",          type=int,
                        help="Vocabulary size for keys in MQAR.")
    parser.add_argument("--n_values",        type=int,
                        help="Vocabulary size for values in MQAR.")
    parser.add_argument("--max_num_pairs",   type=int,
                        help="Padding length (#pairs) for MQAR.")
    parser.add_argument("--unique_keys",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Require unique keys in MQAR.")
    parser.add_argument("--all_queries_for_input",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Return all queries for MQAR (eval).")
    return parser


def update_config_given_args(config, args):
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)
    return config


def setup_experiment(config):
    log_base = config.relative_log_path
    exp_dir  = os.path.join(config.project_path, log_base, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    mprint(f"Log directory: {exp_dir}")
    try:
        cfg_d = {}
        for k, v in dict(config).items():
            if isinstance(v, torch.device):
                cfg_d[k] = str(v)
            elif isinstance(v, (int, float, str, bool, list, tuple, dict)) or v is None:
                cfg_d[k] = v
            else:
                cfg_d[k] = repr(v)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(cfg_d, f, indent=2)
    except Exception as e:
        mprint(f"Warning: failed to save config.json: {e}")
    return None, exp_dir


class PlaceholderTrainer:
    def __init__(self, config, model, train_loader, eval_loader,
                 optimizer, device, logger, exp_log_dir):
        self.config       = config
        self.model        = model
        self.train_loader = train_loader
        self.eval_loader  = eval_loader
        self.optimizer    = optimizer
        self.device       = device
        self.logger       = logger
        self.exp_dir      = exp_log_dir
        self.step         = 0
        mprint(f"Initialized PlaceholderTrainer for {config.dataset}.")

    def train(self):
        mprint(f"Starting training: {self.config.exp_name}...")
        self.model.train()
        total      = self.config.max_train_steps
        log_t      = self.config.log_terminal_every
        eval_e     = self.config.eval_every
        ckpt_e     = self.config.log_ckpt_every
        start_time = time.time()

        while self.step < total:
            try:
                for batch in self.train_loader:
                    if self.step >= total:
                        break
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.optimizer.zero_grad()
                    logits, _ = self.model(x, state=None)

                    # Loss selection
                    if y.dim() == 2:
                        idx = (y.argmax(-1) if y.dtype.is_floating_point and y.shape[-1] > 1 else y)
                        logits_sel = logits[:, -1, :]
                        loss = torch.nn.functional.cross_entropy(logits_sel, idx)

                    elif y.dim() == 3:
                        idx = y.argmax(-1)
                        B, L = idx.shape
                        logits_sel = logits[:, :L, :].reshape(B*L, -1)
                        loss = torch.nn.functional.cross_entropy(logits_sel, idx.reshape(-1))

                    else:
                        B, L = y.shape
                        if logits.shape[1] < L:
                            mprint(f"Warning: skipping batch (logits length {logits.shape[1]} < target length {L})")
                            continue
                        logits_sel = logits[:, :L, :].reshape(B*L, -1)
                        loss = torch.nn.functional.cross_entropy(logits_sel, y.reshape(-1))

                    loss.backward()
                    self.optimizer.step()
                    self.step += 1

                    if self.step % log_t == 0:
                        elapsed = time.time() - start_time
                        mprint(f"Step {self.step}/{total} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")

                    if eval_e > 0 and self.step % eval_e == 0:
                        self.evaluate()
                        self.model.train()

                    if ckpt_e > 0 and self.step % ckpt_e == 0:
                        path = os.path.join(self.exp_dir, f"step_{self.step}.pt")
                        try:
                            torch.save(self.model.state_dict(), path)
                        except Exception as e:
                            mprint(f"Warning: Failed to save checkpoint at step {self.step}: {e}")

            except StopIteration:
                mprint("Training iterator exhausted early.")
                break
            except Exception as e:
                mprint(f"\n--- Error at training step {self.step} ---\n{e}")
                import traceback; traceback.print_exc()
                mprint("Continuing...")
                continue

        mprint("Training finished.")

    def evaluate(self):
        mprint(f"--- Eval at step {self.step} ---")
        self.model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            try:
                for batch in self.eval_loader:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits, _ = self.model(x, state=None)

                    if y.dim() == 2:
                        idx = (y.argmax(-1) if y.dtype.is_floating_point and y.shape[-1] > 1 else y)
                        logits_sel = logits[:, -1, :]
                        loss = torch.nn.functional.cross_entropy(logits_sel, idx, reduction='sum')
                        count += y.size(0)

                    elif y.dim() == 3:
                        idx = y.argmax(-1)
                        B, L = idx.shape
                        logits_sel = logits[:, :L, :].reshape(B*L, -1)
                        loss = torch.nn.functional.cross_entropy(logits_sel, idx.reshape(-1), reduction='sum')
                        count += B * L

                    else:
                        B, L = y.shape
                        if logits.shape[1] < L:
                            continue
                        logits_sel = logits[:, :L, :].reshape(B*L, -1)
                        loss = torch.nn.functional.cross_entropy(logits_sel, y.reshape(-1), reduction='sum')
                        count += B * L

                    total_loss += loss.item()

            except StopIteration:
                mprint("Eval iterator exhausted.")
            except Exception as e:
                mprint(f"\n--- Error during evaluation ---\n{e}")
                import traceback; traceback.print_exc()

        if count > 0:
            mprint(f"Eval Loss: {total_loss / count:.4f}")
        else:
            mprint("No eval items.")
        mprint(f"--- End Eval ---")
def run(config, logger, exp_dir):
    mprint(f"Using device: {config.device}")
    mprint(f"Dataset: {config.dataset}")

    if config.dataset == "bit_parity":
        train_ds = BitParityDatasetIterator(
            batch_size=config.train_batch_size,
            sequence_length=config.train_seq_len,
            pad_sequence_length=config.max_seq_len,
            device=config.device
        )
        eval_ds = BitParityDatasetIterator(
            batch_size=config.eval_batch_size,
            sequence_length=config.eval_seq_len,
            pad_sequence_length=config.max_seq_len,
            device=config.device
        )
        config.num_input_classes = 2
        config.output_size       = 2

    elif config.dataset == "dyck":
        train_ds = DyckDatasetIterator(
            batch_size=config.train_batch_size,
            sequence_length=config.train_seq_len,
            num_parentheses=config.num_parentheses,
            pad_sequence_length=config.max_seq_len,
            depth=config.depth,
            device=config.device
        )
        eval_ds = DyckDatasetIterator(
            batch_size=config.eval_batch_size,
            sequence_length=config.eval_seq_len,
            num_parentheses=config.num_parentheses,
            pad_sequence_length=config.max_seq_len,
            depth=config.depth,
            device=config.device
        )
        config.num_input_classes = config.num_parentheses * 2 + 2
        config.output_size       = 2

    elif config.dataset == "mqar":
        # --- CLAMP train_num_pairs here ---
        min_p, max_p = map(int, config.train_num_pairs.split(','))
        if min_p < 2:
            min_p = 2
        if max_p < min_p:
            max_p = min_p
        config.train_num_pairs = f"{min_p},{max_p}"

        train_ds = MQARDatasetIterator(
            batch_size=config.train_batch_size,
            num_pairs=config.train_num_pairs,
            n_keys=config.n_keys,
            n_values=config.n_values,
            pad_num_pairs=config.max_num_pairs,
            unique_keys=getattr(config, 'unique_keys', False),
            all_queries_for_input=False,
            device=config.device
        )
        eval_ds = MQARDatasetIterator(
            batch_size=config.eval_batch_size,
            num_pairs=config.eval_num_pairs,
            n_keys=config.n_keys,
            n_values=config.n_values,
            pad_num_pairs=config.max_num_pairs,
            unique_keys=getattr(config, 'unique_keys', False),
            all_queries_for_input=getattr(config, 'all_queries_for_input', False),
            device=config.device
        )
        config.num_input_classes = max(config.n_keys, config.n_values) + 2
        config.output_size       = config.n_values + 1

    else:
        raise RuntimeError(f"Unsupported dataset: {config.dataset}")

    mprint(f"Input cls: {config.num_input_classes}, Output sz: {config.output_size}")
    if config.dataset in ("bit_parity", "dyck"):
        mprint(f"Max Seq Len (padding): {config.max_seq_len}")
    else:
        mprint(f"MQAR pad pairs: {config.max_num_pairs}")

    # Model, optimizer, trainer, and start training
    torch.manual_seed(config.seed)
    model = MambaModel(config).to(config.device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    trainer = PlaceholderTrainer(config, model, train_ds, eval_ds,
                                 optim, config.device, logger, exp_dir)
    trainer.train()


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print(f"Usage: python {sys.argv[0]} <config_name> [--dataset …]")
        print(f"Available configs: {config_names}")
        sys.exit(1)

    cfg_name = sys.argv[1]
    if cfg_name not in config_names:
        print(f"Unknown config: {cfg_name}\nAvailable: {config_names}")
        sys.exit(1)

    config = load_config(cfg_name)
    parser = create_parser_based_on_config(config)
    args   = parser.parse_args(sys.argv[2:])
    config = update_config_given_args(config, args)

    # Set device
    config.device = torch.device(args.device or getattr(config, 'device', 'cpu'))

    # Experiment setup
    add_exp_name(config)
    logger, exp_dir = setup_experiment(config)

    try:
        run(config, logger, exp_dir)
    except Exception as e:
        mprint(f"\n--- Fatal error in run() ---\n{e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
