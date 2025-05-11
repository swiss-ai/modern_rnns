import torch
import numpy as np
from typing import Tuple, Dict


class MQARDatasetIterator:
    def __init__(
        self,
        batch_size: int,
        num_pairs: int = 500,
        num_queries: int = 500,
        n_keys: int = 1000,
        n_values: int = 1000,
        pad_num_pairs: int = 500,
        unique_keys: bool = True,
        unique_values: bool = True,
        all_queries_for_input=False,
        device: str = "cpu",
    ):
        """
        A dataset iterator that generates synthetic key-values pair sequences
        and keys' associated values.

        Args:
            batch_size (int): Number of sequences per batch.
            num_pairs (int): Number of key-value pairs to be generated.
            n_keys (int): The number of possible distinct keys generated.
            n_values (int): The number of possible distinct values generated.
            unique_keys (bool): True if the every generated sequence needs to
                                have unique keys.
            all_queries_for_input (bool): True if instead of a batch of key
                            value pair sequences with an input, it returns all
                            the possible queries for a given input
            device (str): Device to store the tensors ('cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.min_num_pairs, self.max_num_pairs = self._parse_num_pairs(num_pairs)
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.n_keys = n_keys
        self.n_values = n_values
        self.pad_num_pairs = pad_num_pairs
        self.unique_keys = unique_keys
        self.unique_values = unique_values
        self.all_queries_for_input = all_queries_for_input
        self.device = device

        self.pipe_token = n_keys + n_values
        self.query_token = self.pipe_token + 1

        if self.unique_keys:
            assert n_keys >= self.min_num_pairs, (
                "The number of different keys is smaller than the minimum "
                "length of the sequence"
            )
        if self.unique_values:
            assert n_values >= self.min_num_pairs, (
                "The number of different values is smaller than the minimum "
                "length of the sequence"
            )

        self.key_indexes = list(range(self.n_keys))
        self.value_indexes = list(range(self.n_keys, self.n_keys + self.n_values))

    def pretty_print(self, tensor):
        tensor = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
        for i, element in enumerate(tensor):
            if element == self.pipe_token:
                s = "="
            elif element == self.query_token:
                s = "?, "
            else:
                s = str(element.item())
                if (i > 0 and tensor[i - 1] == self.pipe_token):
                    s += ", "
            print(s, end="")
        print()

    def pretty_print_preds(self, tensor, preds):
        tensor = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
        preds = preds.cpu().numpy() if preds.is_cuda else preds.numpy()
        for i, element in enumerate(preds):
            print(str(element.item()), end="")
            if tensor[i] == self.query_token or (i > 0 and tensor[i - 1] == self.pipe_token):
                print(", ", end="")
        print()

    def _parse_num_pairs(self, num_pairs):
        values = tuple(map(int, num_pairs.split(",")))
        if len(values) != 2:
            raise ValueError(
                "The number of pairs must be in the format 'int,int' (e.g., 5,10)."
            )
        if values[0] > values[1]:
            raise ValueError(
                "The first element of num_pairs must be less than or equal to the second."
            )
        if values[0] <= 0 or values[1] <= 0:
            raise ValueError("Both elements of num_pairs must be positive integers.")
        return values[0], values[1]

    def __iter__(self):
        return self

    def __get_ground_truth(self, kv_sequence):
        """
        Returns a dictionary with the last value associated with every key
        in the flat key-value sequence.
        """
        ground_truth = dict()
        for i in range(0, len(kv_sequence), 3):
            key = kv_sequence[i].item()
            value = kv_sequence[i + 2].item()
            ground_truth[key] = value
        return ground_truth

    def __generate_sequence(self, num_pairs, num_queries):
        keys = np.random.choice(
            self.key_indexes, size=num_pairs, replace=not self.unique_keys
        )
        values = np.random.choice(
            self.value_indexes, size=num_pairs, replace=not self.unique_values
        )

        # must start with a key-value pair
        sequence = np.array([keys[0], self.pipe_token, values[0]], dtype=np.int32)
        targets = np.array([0, 0, 0], dtype=np.int32)
        ground_truth = {keys[0]: values[0]}
        num_keys_set = 1
        num_queries_set = 0

        while num_keys_set < num_pairs or num_queries_set < num_queries:
            if num_queries_set == num_queries or (
                num_keys_set < num_pairs and np.random.rand() <= 0.5
            ):
                sequence = np.concatenate(
                    (
                        sequence,
                        [keys[num_keys_set], self.pipe_token, values[num_keys_set]],
                    ),
                    axis=0,
                )
                targets = np.concatenate((targets, [0, 0, 0]), axis=0)
                ground_truth[keys[num_keys_set]] = values[num_keys_set]
                num_keys_set += 1
            else:
                query_key = np.random.choice(
                    keys[:num_keys_set], size=1, replace=False
                )[0]
                query_value = ground_truth[query_key]
                sequence = np.concatenate(
                    (sequence, [query_key, self.query_token]), axis=0
                )
                targets = np.concatenate((targets, [0, query_value]), axis=0)
                num_queries_set += 1

        sequence_tensor = torch.tensor(sequence, dtype=torch.long).to(self.device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(self.device)
        return sequence_tensor, target_tensor

    def __generate_batch(self):
        batch_sequences = []
        batch_targets = []
        num_pairs = torch.randint(
            self.min_num_pairs, self.max_num_pairs + 1, (1,)
        ).item()

        for _ in range(self.batch_size):
            kv_sequence, ground_truth = self.__generate_sequence(
                num_pairs, self.num_queries
            )
            batch_sequences.append(kv_sequence)
            batch_targets.append(ground_truth)

        batch_sequences = torch.stack(batch_sequences)
        batch_targets = torch.stack(batch_targets)
        batch_sequences = torch.nn.functional.pad(
            batch_sequences, (0, (self.pad_num_pairs - num_pairs) * 2), value=0
        )
        return batch_sequences, batch_targets

    def __generate_all_queries(self):
        kv_sequence = self.__generate_sequence()
        ground_truth = self.__get_ground_truth(kv_sequence)
        batch_sequences = []
        batch_targets = []

        for query_key in ground_truth.keys():
            extended_sequence = torch.cat(
                [kv_sequence, torch.tensor([query_key], device=self.device)]
            )

            target_value = ground_truth[query_key]
            target_one_hot = torch.zeros(self.n_values + 1, device=self.device)
            target_one_hot[target_value] = 1

            batch_sequences.append(extended_sequence)
            batch_targets.append(target_one_hot)

        return torch.stack(batch_sequences), torch.stack(batch_targets)

    def __next__(self) -> Tuple[torch.Tensor, Dict[int, int]]:
        if self.all_queries_for_input:
            return self.__generate_all_queries()

        return self.__generate_batch()


# Example usage
if __name__ == "__main__":
    dataset = MQARDatasetIterator(
        batch_size=4,
        num_pairs=5,
        n_keys=3,
        n_values=3,
        unique_keys=False,
        all_queries_for_input=False,
        device="cpu",
    )
    for _ in range(3):
        x, y = next(dataset)
        print("Input:", x)
        print("Output:", y)
        print()
