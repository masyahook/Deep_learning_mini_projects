import math
import random
import torch


def batchify(dataset, batch_size=1, shuffle=True):
    X, y = dataset
    order = list(range(len(y)))

    if shuffle:
        random.shuffle(order)

    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]

        yield X[batch_idx], y[batch_idx]


def generate_data(size):
    x = torch.rand(size, 2)
    y = (((x - 0.5) ** 2).sum(dim=1) <= 1 / (2 * math.pi)).int()

    return x, y
