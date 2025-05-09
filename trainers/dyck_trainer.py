import os
import torch
from tqdm import tqdm

from common_lib import parallel_utils
from sklearn.metrics import precision_score, recall_score


class DyckTrainer:
    def __init__(
        self,
        config,
        model,
        train_loader,
        eval_loader,
        optimizer,
        device,
        max_steps=40000,
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
        self.criterion_no_reduction = torch.nn.CrossEntropyLoss(reduction='none')

    def train(self):
        self.model.train()
        step = 0
        state = self._init_state()
        total_correct = 0
        total_samples = 0

        while step < self.max_steps:
            inputs, targets = next(self.train_loader)

            state = self._init_state()
            logits, state = self.model(inputs, state)
            targets = torch.argmax(targets, dim=2)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            masked_loss = self.criterion_no_reduction(logits.view(-1, logits.size(-1)), targets.view(-1)).view_as(targets)
            masked_loss = (targets.float() * masked_loss)

            averaged_loss = masked_loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=2)

            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0) * targets.size(1)

            # Detach state so it doesn’t backpropagate through the whole history
            state = self._detach_state(state)

            if step % 100 == 0:
                accuracy = total_correct / total_samples

                # Flatten predictions and targets for precision/recall
                flat_preds = preds.view(-1).cpu().numpy()
                flat_targets = targets.view(-1).cpu().numpy()

                precision = precision_score(flat_targets, flat_preds, average='macro', zero_division=0)
                recall = recall_score(flat_targets, flat_preds, average='macro', zero_division=0)

                print(f"[Step {step}] Train loss: {loss.item():.4f}, Masked loss: {averaged_loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

                if self.logger:
                    self.logger.log({
                        "train/loss": loss.item(),
                        "train/masked_loss": averaged_loss.item(),
                        "train/accuracy": accuracy,
                        "train/precision": precision,
                        "train/recall": recall
                    }, step)

                total_correct = 0
                total_samples = 0

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
        all_preds = []
        all_labels = []
        state = self._init_state()

        with torch.no_grad():
            for _ in range(10):  # Evaluate on 10 batches
                inputs, targets = next(self.eval_loader)
                labels = torch.argmax(targets, dim=2)
                state = self._init_state()

                logits, state = self.model(inputs, state)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
                masked_loss = self.criterion_no_reduction(logits.view(-1, logits.size(-1)), labels.view(-1)).view_as(labels)
                masked_loss = (labels.float() * masked_loss)

                averaged_loss = masked_loss.mean()

                preds = torch.argmax(logits, dim=2)

                total_correct += (preds == labels).sum().item()
                total_samples += targets.size(0) * targets.size(1)

                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())

        avg_loss = total_loss / 10
        accuracy = total_correct / total_samples
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"[Eval @ Step {step}] Loss: {avg_loss:.4f}, Masked loss: {averaged_loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        if self.logger:
            self.logger.log({
                "eval/loss": avg_loss,
                "eval/masked_loss": averaged_loss.item(),
                "eval/accuracy": accuracy,
                "eval/precision": precision,
                "eval/recall": recall
            }, step)

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
        if isinstance(state, tuple):
            state[0].detach()
            state[1].detach()
            return
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
