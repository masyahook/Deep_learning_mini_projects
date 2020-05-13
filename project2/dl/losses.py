import torch
from .module import Module


class LossMSE(Module):
    def forward(self, input, target):
        self.diff = input - target
        return (self.diff ** 2).sum() / input.shape[0]

    def backward(self):
        return 2 * self.diff / self.diff.shape[0]


class LossBCE(Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.EPS = eps

    def forward(self, input, target):
        self.input = torch.clamp(input, self.EPS, 1 - self.EPS)
        self.target = target

        return (-target * self.input.log() - (1 - target) * (1 - self.input).log()).sum() / self.input.shape[0]

    def backward(self):
        return (-self.target / self.input + (1 - self.target) / (1 - self.input)) / self.input.shape[0]
