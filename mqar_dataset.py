import torch
import numpy as np
from typing import Tuple, Dict

class MQARDatasetIterator:
    def __init__(self, batch_size: int, num_pairs: int = 500, 
                n_keys: int = 1000, 
                n_values: int = 1000, 
                unique_keys: bool = True, 
                all_queries_for_input = False,
                device: str ="cpu"):
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
        self.num_pairs = num_pairs
        self.n_keys = n_keys
        self.n_values = n_values
        self.unique_keys = unique_keys
        self.all_queries_for_input = all_queries_for_input
        self.device = device

        if self.unique_keys:
            assert n_keys > num_pairs, (
                "The number of different keys is smaller than the length of the"
                "sequence"
            )
        
        self.key_indexes = list(range(self.n_keys))
        self.value_indexes = list(range(self.n_keys, self.n_keys + self.n_values))

    def __iter__(self):
        return self
    
    def __get_ground_truth(self, kv_sequence):
        """
            Returns a dictionary with the last value associated with every key
            in the key-value pair sequence.
        """
        ground_truth = dict()
        keys, values = kv_sequence

        for i in range(keys.shape[0]):
            key = keys[i].item()
            value = values[i]
            ground_truth[key] = value  

        return ground_truth
    
    def __generate_sequence(self):
        keys = np.random.choice(
            self.key_indexes,
            size = self.num_pairs,
            replace = not self.unique_keys
        )

        values = np.random.choice(
            self.value_indexes,
            size = self.num_pairs,
            replace = True
        )

        keys_tensor = torch.tensor(keys).long()
        values_tensor = torch.tensor(values).long()

        values_onehot = torch.nn.functional.one_hot(
            values_tensor - self.n_keys,
            num_classes=self.n_values
        ).float()

        return (keys_tensor.to(self.device), values_onehot.to(self.device))

    def __generate_sample(self):
        kv_pairs = self.__generate_sequence()
        return kv_pairs, self.__get_ground_truth(kv_pairs) 
    
    def __generate_batch(self):
        """
        Generates a new batch of synthetic mqar sequences and the associated 
        values with each pair.
        """
        batch_kv = []
        batch_targets = []

        for _ in range(self.batch_size):
            kv_pairs, ground_truth = self.__generate_sample() 
            
            keys, values = kv_pairs

            query_index = np.random.randint(0, keys.shape[0])
            query_key = keys[query_index].unsqueeze(0)
            true_value = ground_truth[query_key.item()].to(self.device)

            query_value = torch.zeros((1, self.n_values), device = self.device)

            keys_with_query = torch.cat([keys, query_key], dim = 0).to(self.device)
            values_with_query = torch.cat([values, query_value], dim = 0).to(self.device)

            batch_kv.append((keys_with_query, values_with_query))
            batch_targets.append(true_value)
        
        return batch_kv, batch_targets
    
    def __generate_all_queries(self):
        """
        Generate a batch of syntethic mqar queries, where all have the same
        input, but different key queries.
        """
        kv_pairs = self.__generate_sequence()
        ground_truth = self.__get_ground_truth(kv_pairs)
        keys, values = kv_pairs

        batch_kv = []
        batch_targets = []

        for query_key in ground_truth.keys():
            query_value = torch.zeros((1, self.n_values), device = self.device)

            keys_with_query = torch.cat([keys, torch.tensor([query_key])], dim = 0).to(self.device)
            values_with_query = torch.cat([values, query_value], dim = 0).to(self.device)

            batch_kv.append((keys_with_query, values_with_query))
            batch_targets.append(ground_truth[query_key].to(self.device))
        
        return batch_kv, batch_targets
    
    def __next__(self) -> Tuple[torch.Tensor, Dict[int, int]]:
        if self.all_queries_for_input:
            return self.__generate_all_queries()

        return self.__generate_batch()

# Example usage
if __name__ == "__main__":
    dataset = MQARDatasetIterator(batch_size=4, num_pairs = 5, n_keys = 3, n_values = 3, unique_keys = False, all_queries_for_input = True, device="cpu")
    for _ in range(3):
        x, y = next(dataset)
        print("Input:", x)
        print("Output:", y)
        print()

