import math
import random
import torch


def batchify(dataset, batch_size=1, shuffle=True):
    """
        Helper function that splits the dataset in batches and shuffles them.
        
        Parameters
        ----------
        dataset : tuple of Torch tensors
            Pair of (feature set, label set) that needs to be splitted into batches
        batch_size : int
            The size of the batch (by default 1)
        shuffle : boolean
            True if shuffle the samples, False if not
                
        Returns
        -------
        Generator of splitted dataset
            Dataset splitted into batches
    """
    X, y = dataset
    order = list(range(len(y)))

    # Shuffling the samples
    if shuffle:
        random.shuffle(order)

    # Splitting dataset into batches
    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]

        yield X[batch_idx], y[batch_idx]


def generate_data(size):
    """
        Helper function that generates a toy dataset.
        
        Parameters
        ----------
        size : int
            The size of the dataset
                
        Returns
        -------
        Tuple of Torch tensors
            Pair of (feature set, label set)
    """
    x = torch.rand(size, 2)
    y = (((x - 0.5) ** 2).sum(dim=1) <= 1 / (2 * math.pi)).int()

    return x, y
