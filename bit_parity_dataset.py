import torch

from common_lib.dataset_utils import parse_sequence_length


class BitParityDatasetIterator:
    def __init__(self, batch_size, sequence_length, device="cpu"):
        """
        A dataset iterator that generates synthetic binary sequences and their parity labels.

        Args:
            batch_size (int): Number of sequences per batch.
            sequence_length (string): String concatenating the min and max lengths of the sequence.
            device (str): Device to store the tensors ('cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.min_seq_len, self.max_seq_len = parse_sequence_length(
            sequence_length
        )
        self.device = device



    def __iter__(self):
        return self

    def __next__(self):
        """Generates a new batch of synthetic binary sequences and their parity labels."""
        # Generate random binary sequences
        sequence_length = torch.randint(
            self.min_seq_len, self.max_seq_len + 1, (1,)
        ).item()
        batch_x = torch.randint(
            0, 2, (self.batch_size, sequence_length), dtype=torch.int64
        )

        # Compute parity (sum of 1s mod 2) and convert to one-hot
        # parity_labels = batch_x.sum(dim=1) % 2
        # batch_y = torch.nn.functional.one_hot(parity_labels, num_classes=2).to(
        #     dtype=torch.int64
        # )

        # holds partial sums (1 if partial sum is even, else 0)
        # parity_labels = [batch_x[:, :i + 1].sum(dim=1) % 2 == 0 for i in range(self.sequence_length)]
        # batch_y = torch.stack(parity_labels, dim=1).int()

        # Compute parity labels without using cumsum
        parity_labels = [
            batch_x[:, : i + 1].sum(dim=1) % 2 == 0
            for i in range(sequence_length)
        ]
        batch_y = torch.stack(parity_labels, dim=1).to(dtype=torch.float)
        batch_y = torch.stack([1 - batch_y, batch_y], dim=2)

        return batch_x.to(self.device), batch_y.to(self.device)


# Example usage
if __name__ == "__main__":
    dataset = BitParityDatasetIterator(
        batch_size=1, sequence_length="3,3", device="cpu"
    )
    for _ in range(3):  # Generate 3 batches
        x, y = next(dataset)
        print("Input:", x)
        print("Output:", y)
        print()
