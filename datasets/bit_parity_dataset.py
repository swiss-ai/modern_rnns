import torch


class BitParityDatasetIterator:
    def __init__(self, batch_size, sequence_length, pad_sequence_length, device="cpu"):
        """
        A dataset iterator that generates synthetic binary sequences and their parity labels.

        Args:
            batch_size (int): Number of sequences per batch.
            sequence_length (string): String concatenating the min and max lengths of the sequence.
            pad_sequence_length (int): Maximum possible length of the binary sequence.
                It pads all sequences to this length.
            device (str): Device to store the tensors ('cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.min_seq_len, self.max_seq_len = self._parse_sequence_length(
            sequence_length
        )
        self.pad_sequence_length = pad_sequence_length
        if self.pad_sequence_length < self.max_seq_len:
            raise ValueError(
                f"The total padded sequence length [{self.pad_sequence_length}] "
                "must be greater than or equal to the max sequence length [{self.max_seq_len}]."
            )
        self.device = device

    def _parse_sequence_length(self, sequence_length):
        values = tuple(map(int, sequence_length.split(",")))
        if len(values) != 2:
            raise ValueError(
                "Sequence length must be in the format 'int,int' (e.g., 5,10)."
            )
        if values[0] > values[1]:
            raise ValueError(
                "The first element of sequence_length must be less than or equal to the second."
            )
        if values[0] <= 0 or values[1] <= 0:
            raise ValueError(
                "Both elements of sequence_length must be positive integers."
            )
        return values[0], values[1]

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
        batch_x = torch.nn.functional.pad(
            batch_x, (0, self.pad_sequence_length - sequence_length), value=0
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
            for i in range(self.pad_sequence_length)
        ]
        batch_y = torch.stack(parity_labels, dim=1).to(dtype=torch.float)
        batch_y = torch.stack([1 - batch_y, batch_y], dim=2)

        return batch_x.to(self.device), batch_y.to(self.device)


# Example usage
if __name__ == "__main__":
    dataset = BitParityDatasetIterator(
        batch_size=8, sequence_length="10,10", pad_sequence_length=10, device="cpu"
    )
    for _ in range(3):  # Generate 3 batches
        x, y = next(dataset)
        print("Input:", x)
        print("Output:", y)
        print()
