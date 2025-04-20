import torch
from tqdm import tqdm  # Optional for adding progress bars

class MQARTrainer:
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

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits, state = self.model(inputs, state)
            labels = torch.argmax(targets, dim = 1)
            loss = self.criterion(logits[:, -1], labels)  

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(logits[:, -1], dim=1)
            _, true_values = torch.max(targets, dim=1)
            correct = (predicted == true_values).sum().item()

            state = self._detach_state(state)

            if step % 100 == 0:
                accuracy = correct / targets.size(0)
                print(f"[Step {step}] Train loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
                if self.logger:
                    self.logger.log({"train/loss": loss.item(), "train/accuracy": accuracy}, step)

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
                state = self._init_state()

                logits, state = self.model(inputs, state)
                labels = torch.argmax(targets, dim = 1)
                loss = self.criterion(logits[:, -1], labels)  
                total_loss += loss.item()

                _, predicted = torch.max(logits[:, -1], dim=1)
                _, true_values = torch.max(targets, dim=1)

                total_correct += (predicted == true_values).sum().item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / 10 
        accuracy = total_correct / total_samples
        print(f"[Eval @ Step {step}] Loss: {avg_loss:.4f} , Accuracy: {accuracy:.4f}")

        if self.logger:
            self.logger.log({"eval/loss": avg_loss, "eval/accuracy": accuracy}, step)

        self.model.train()

    def _init_state(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.get_init_state(batch_size=self.train_loader.batch_size, device=self.device)
        else:
            return self.model.get_init_state(batch_size=self.train_loader.batch_size, device=self.device)

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