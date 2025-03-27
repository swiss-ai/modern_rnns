import torch


class BitParityDatasetIterator:
    def __init__(self, batch_size, sequence_length, device="cpu"):
        """
        A dataset iterator that generates synthetic binary sequences and their parity labels.

        Args:
            batch_size (int): Number of sequences per batch.
            sequence_length (int): Length of each binary sequence.
            device (str): Device to store the tensors ('cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        """Generates a new batch of synthetic binary sequences and their parity labels."""
        # Generate random binary sequences
        batch_x = torch.randint(
            0, 2, (self.batch_size, self.sequence_length), dtype=torch.int64
        )

        # Compute parity (sum of 1s mod 2) and convert to one-hot
        parity_labels = batch_x.sum(dim=1) % 2
        batch_y = torch.nn.functional.one_hot(parity_labels, num_classes=2).to(
            dtype=torch.int64
        )

        return batch_x.to(self.device), batch_y.to(self.device)


# Example usage
if __name__ == "__main__":
    dataset = BitParityDatasetIterator(batch_size=8, sequence_length=10, device="cpu")
    for _ in range(3):  # Generate 3 batches
        x, y = next(dataset)
        print("Input:", x)
        print("Output:", y)
        print()
