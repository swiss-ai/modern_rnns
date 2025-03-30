import torch
from tqdm import tqdm  # Optional for adding progress bars

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

        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        step = 0
        state = self._init_state()

        while step < self.max_steps:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                
                logits, state = self.model(inputs, state)
                
                loss = self.criterion(logits[:, -1], targets)  

                loss.backward()
                self.optimizer.step()

                state = self._detach_state(state)

                if step % 100 == 0:
                    print(f"[Step {step}] Train loss: {loss.item():.4f}")
                    if self.logger:
                        self.logger.log({"train/loss": loss.item()}, step)

                if step % self.eval_every == 0 and step > 0:
                    self.evaluate(step)

                step += 1
                if step >= self.max_steps:
                    break

    def evaluate(self, step):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        state = self._init_state()

        with torch.no_grad():
            for _ in range(10):
                inputs, targets = next(self.eval_loader)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits, state = self.model(inputs, state)
                
                loss = self.criterion(logits[:, -1], targets)
                total_loss += loss.item()

        avg_loss = total_loss / 10 
        print(f"[Eval @ Step {step}] Loss: {avg_loss:.4f}")

        if self.logger:
            self.logger.log({"eval/loss": avg_loss}, step)

        self.model.train()

    def _init_state(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.get_init_state(batch_size=self.train_loader.batch_size, device=self.device)
        else:
            return self.model.get_init_state(batch_size=self.train_loader.batch_size, device=self.device)

    def _detach_state(self, state):
        if state is None:
            return None
        return (
            {k: v.detach() if v is not None else None for k, v in state.items()}
            if isinstance(state, dict)
            else state.detach()
        )
