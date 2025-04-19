import torch
import numpy as np


class DyckDatasetIterator:
    def __init__(
        self, batch_size, sequence_length, num_parentheses, pad_sequence_length, depth, device="cpu"
    ):
        """
        A dataset iterator that generates synthetic Dyck langauge sequences and
        their correctness labels.

        Args:
            batch_size (int): Number of sequences per batch.
            sequence_length (int): Length of each Dyck sequence.
            num_parentheses (int): The number of possible distinct parentheses.
            depth (int): The maximum depth of opened parentheses.
            device (str): Device to store the tensors ('cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.min_seq_len, self.max_seq_len = self._parse_sequence_length(
            sequence_length
        )
        self.pad_sequence_length = pad_sequence_length
        self.num_parentheses = num_parentheses
        self.depth = depth
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

    def __is_opening_paranthesis(self, paranthesis):
        return paranthesis < self.num_parentheses

    def __get_closing_paranthesis(self, paranthesis):
        return paranthesis + self.num_parentheses

    def __one_hot_label(self, valid):
        return [0.0, 1.0] if valid else [1.0, 0.0]

    def __compute_labels(self, sequence):
        # Computes the correctness of
        labels = []
        stack = []
        not_matching_closed = True

        for paranthesis in sequence:
            if self.__is_opening_paranthesis(paranthesis):
                stack.append(paranthesis)
                labels.append(self.__one_hot_label(False))
                continue

            if len(stack) == 0:
                not_matching_closed = False
                labels.append(self.__one_hot_label(False))
                continue

            if self.__get_closing_paranthesis(stack[-1]) == paranthesis:
                stack.pop()
                labels.append(
                    self.__one_hot_label(len(stack) == 0 and not_matching_closed)
                )
                continue

            not_matching_closed = False
            labels.append(self.__one_hot_label(False))

        return labels

    def __shuffle_sample(self, sequence, labels):
        valid_positions = [i for i, x in enumerate(labels[:-1]) if x == [0.0, 1.0]]
        if len(valid_positions) == 0:
            np.random.shuffle(sequence)
            shuffled_sequence_labels = self.__compute_labels(sequence)
            return sequence, shuffled_sequence_labels

        start_shuffle_position = np.random.choice(valid_positions)

        shuffled_sequence = sequence[start_shuffle_position + 1 :]
        np.random.shuffle(shuffled_sequence)
        shuffled_sequence_labels = self.__compute_labels(shuffled_sequence)

        sequence[start_shuffle_position + 1 :] = shuffled_sequence
        labels[start_shuffle_position + 1 :] = shuffled_sequence_labels

        return sequence, labels

    def __generate_sequence(self, length):
        sequence = []
        stack = []
        labels = []

        for _ in range(length):
            if length - len(sequence) == len(stack):
                while len(stack):
                    sequence.append(stack.pop())
                    labels.append(self.__one_hot_label(len(stack) == 0))
                break

            if len(stack) == self.depth or (len(stack) and np.random.rand() <= 0.5):
                sequence.append(stack.pop())
                labels.append(self.__one_hot_label(len(stack) == 0))
                continue

            open_paranthesis = np.random.randint(self.num_parentheses)
            sequence.append(open_paranthesis)
            stack.append(self.__get_closing_paranthesis(open_paranthesis))
            labels.append(self.__one_hot_label(False))

        #print(sequence, labels)
        return sequence, labels

    def __generate_sample(self, length):
        sequence, labels = self.__generate_sequence(length)

        if np.random.rand() <= 0.5:
            sequence, labels = self.__shuffle_sample(sequence=sequence, labels=labels)

        return sequence, labels

    def __next__(self):
        batch_sequences = []
        batch_labels = []
        if self.min_seq_len // 2 == (self.max_seq_len + 1) // 2:
            sequence_length = (self.min_seq_len // 2) * 2
        else:
            sequence_length = np.random.randint(self.min_seq_len // 2, (self.max_seq_len + 1) // 2) * 2

        for _ in range(self.batch_size):
            sequence, labels = self.__generate_sample(sequence_length)
            batch_sequences.append(sequence)
            batch_labels.append(labels)

        batch_sequences = torch.tensor(batch_sequences)
        batch_labels = torch.tensor(batch_labels)
        batch_sequences = torch.nn.functional.pad(
            batch_sequences, (0, self.pad_sequence_length - sequence_length), value=self.num_parentheses * 2 + 1
        )
        batch_labels = torch.nn.functional.pad(
            batch_labels, (0, self.pad_sequence_length - sequence_length), value=1
        )
        return batch_sequences.to(self.device), batch_labels.to(self.device)


# Example usage
if __name__ == "__main__":
    dataset = DyckDatasetIterator(
        batch_size=2, sequence_length=8, num_parentheses=3, depth=4, device="cpu"
    )
    for _ in range(3):  # Generate 3 batches
        x, y = next(dataset)
        print("Input:", x)
        print("Output:", y)
        print()
