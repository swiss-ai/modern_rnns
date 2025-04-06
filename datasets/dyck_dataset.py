import torch
import numpy as np

class DyckDatasetIterator:
    def __init__(self, batch_size, sequence_length, no_parantheses, depth, device="cpu"):
        """
        A dataset iterator that generates synthetic Dyck langauge sequences and 
        their correctness labels.

        Args:
            batch_size (int): Number of sequences per batch.
            sequence_length (int): Length of each Dyck sequence.
            no_parantheses (int): The number of possible distinct parentheses.
            depth (int): The maximum depth of opened parentheses.
            device (str): Device to store the tensors ('cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.no_pharanteses = no_parantheses 
        self.depth = depth
        self.device = device

    def __iter__(self):
        return self
    
    def __get_closing_paranthesis(self, paranthesis):
        return paranthesis + self.no_pharanteses
    
    def __one_hot_label(self, valid):
        return [1.0, 0.0] if valid else [0.0, 1.0]
    
    def __generate_sequence(self, length):
        sequence = []
        stack = []
        labels = []

        for _ in range(length):
            if self.sequence_length - len(sequence) == len(stack):
                while len(stack):
                    sequence.append(stack.pop())
                    labels.append(self.__one_hot_label(len(stack) == 0))
                break

            if len(stack) == self.depth or (len(stack) and np.random.rand() <= 0.5):
                sequence.append(stack.pop())
                labels.append(self.__one_hot_label(len(stack) == 0))
                continue
            
            open_paranthesis = np.random.randint(self.no_pharanteses) 
            sequence.append(open_paranthesis)
            stack.append(self.__get_closing_paranthesis(open_paranthesis))
            labels.append(self.__one_hot_label(False))
            continue

        return sequence, labels 
    
    def __next__(self):
        batch_sequences = []
        batch_labels = []

        for _ in range(self.batch_size):
            sequence, labels = self.__generate_sequence(self.sequence_length)
            batch_sequences.append(sequence)
            batch_labels.append(labels)

        return torch.tensor(batch_sequences).to(self.device), \
                    torch.tensor(batch_labels).to(self.device)

# Example usage
if __name__ == "__main__":
    dataset = DyckDatasetIterator(batch_size=4, sequence_length=20, no_parantheses = 6, depth = 4, device="cpu")
    for _ in range(3):  # Generate 3 batches
        x, y = next(dataset)
        print("Input:", x)
        print("Output:", y)
        print()