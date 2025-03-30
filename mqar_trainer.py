import torch
import torch.nn.functional as F

class MQARTrainer:
    def __init__(
        self,
        model,
        train_loader,
        eval_loader,
        optimizer,
        device,
        max_steps=50000,
        eval_every=1000,
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

        self.criterion = torch.nn.MSELoss()

    def train(self):
        self.model.train()
        step = 0
        state = self._init_state()

        while step < self.max_steps:
            batch, targets = next(self.train_loader)
            total_loss = 0.0

            self.optimizer.zero_grad()

            for (keys, values), target in zip(batch, targets):
                x = [keys.to(self.device), values.to(self.device)]
                target = target.to(self.device)
                preds, state = self.model(x, state)
                pred_query = preds[0, -1]  # [value_size]

                loss = 0.5 * ((pred_query - target) ** 2).sum()
                loss.backward()
                total_loss += loss.item()

            self.optimizer.step()
            state = self._detach_state(state)

            if step % 100 == 0:
                avg_loss = total_loss / len(batch)
                print(f"[Step {step}] Train loss: {avg_loss:.4f}")
                if self.logger:
                    self.logger.log({"train/loss": avg_loss}, step)

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
            for _ in range(10):
                batch, targets = next(self.eval_loader)

                for (keys, values), target in zip(batch, targets):
                    x = [keys.to(self.device), values.to(self.device)]
                    target = target.to(self.device)
                    preds, state = self.model(x, state)
                    pred_query = preds[0, -1]  # [value_size]

                    loss = 0.5 * self.criterion(pred_query, target) 
                    total_loss += loss.item()

                    pred_index = pred_query.argmax()
                    true_index = target.argmax()
                    total_correct += (pred_index == true_index).item()
                    total_samples += 1

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
