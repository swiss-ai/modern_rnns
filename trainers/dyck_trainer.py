import os
import torch
from tqdm import tqdm

from common_lib import parallel_utils


class DyckTrainer:
    def __init__(
        self,
        config,
        model,
        train_loader,
        eval_loader,
        optimizer,
        device,
        max_steps=50000,
        eval_every=1000,
        logger=None,
    ):
        self.c = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        step = 0
        state = self._init_state()

        while step < self.max_steps:
            inputs, targets = next(self.train_loader)

            logits, state = self.model(inputs, state)

            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Detach state so it doesn’t backpropagate through the whole history
            state = self._detach_state(state)

            if step % 100 == 0:
                print(f"[Step {step}] Train loss: {loss.item():.4f}")
                if self.logger:
                    self.logger.log({"train/loss": loss.item()}, step)

            if step % self.eval_every == 0 and step > 0:
                self.evaluate(step)

            step += 1

            if (
                parallel_utils.is_main_process()
                and self.c.log_ckpt_every > 0
                and step % self.c.log_ckpt_every == 0
                and step > 0
            ):
                self.save_checkpoint(self.logger, step)

        # Final validation run and checkpoint
        self.evaluate(step=step)
        if parallel_utils.is_main_process():
            self.save_checkpoint(self.logger, step)

    def evaluate(self, step):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        state = self._init_state()

        with torch.no_grad():
            for _ in range(10):  # Evaluate on 10 batches
                inputs, targets = next(self.eval_loader)
                logits, state = self.model(inputs, state)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=2)

                labels = torch.argmax(targets, dim=2)
                total_correct += (preds == labels).sum().item()
                total_samples += targets.size(0) * targets.size(1)

        avg_loss = total_loss / 10
        accuracy = total_correct / total_samples
        print(f"[Eval @ Step {step}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        if self.logger:
            self.logger.log({"eval/loss": avg_loss, "eval/accuracy": accuracy}, step)

        self.model.train()

    def _init_state(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.get_init_state(
                batch_size=self.train_loader.batch_size, device=self.device
            )
        else:
            return self.model.get_init_state(
                batch_size=self.train_loader.batch_size, device=self.device
            )

    def _detach_state(self, state):
        if state is None:
            return None
        return (
            {k: v.detach() if v is not None else None for k, v in state.items()}
            if isinstance(state, dict)
            else (
                [(s[0].detach(), s[1].detach()) for s in state]
                if isinstance(state, list)
                else state.detach()
            )
        )

    def save_checkpoint(self, logger, step):
        """Saves a checkpoint of the current model to disk."""

        def _save_checkpoint(path):
            # create folder if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            # remove previous log file there is one
            file = os.path.join(path, "model.pt")
            if os.path.exists(file):
                os.remove(file)

            # Write checkpoint
            with open(file, "wb") as f:
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": self.model.state_dict(),
                        "opt_state_dict": self.optimizer.state_dict(),
                    },
                    f,
                )
            print(f"Checkpoint written at step {step} to:\n{file}")

        if logger.use_tb:
            ckpt_path = os.path.join(logger.log_path, "checkpoints")
            _save_checkpoint(ckpt_path)

        if logger.use_wandb:
            ckpt_path = os.path.join(logger.wandb_run_dir, "checkpoints")
            _save_checkpoint(ckpt_path)
