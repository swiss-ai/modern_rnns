import torch
from tqdm import tqdm


class BitParityTrainer:
    def __init__(
        self,
        model,
        train_loader,
        eval_loader,
        optimizer,
        device,
        max_steps=1000,
        eval_every=100,
        logger=None,
    ):
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
            loss = self.criterion(logits, targets.argmax(dim=-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Detach state so it doesn’t backpropagate through the whole history
            state = self._detach_state(state)

            if step % 10 == 0:
                print(f"[Step {step}] Train loss: {loss.item():.4f}")
                if self.logger:
                    self.logger.log({"train/loss": loss.item()}, step)

            if step % self.eval_every == 0 and step > 0:
                self.evaluate(step)

            step += 1

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
                loss = self.criterion(logits, targets.argmax(dim=-1))
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                labels = torch.argmax(targets, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += targets.size(0)

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
            else state.detach()
        )
