import torch
from .module import Module


class Linear(Module):
    def __init__(self, n_in, n_out):
        super().__init__()

        stdv = 1. / n_in ** 0.5
        self.W = (torch.rand(n_out, n_in) - 0.5) * 2 * stdv
        self.b = (torch.rand(n_out) - 0.5) * 2 * stdv

        self.gradW = torch.zeros_like(self.W)
        self.gradb = torch.zeros_like(self.b)

    def forward(self, input):
        self.input = input

        return input @ self.W.T + self.b

    def backward(self, gradwrtoutput):
        self.gradW += (self.input[:, None, :] * gradwrtoutput[:, :, None]).sum(axis=0)
        self.gradb += gradwrtoutput.sum(axis=0)

        return gradwrtoutput @ self.W

    def param(self):
        return [(self.W, self.gradW), (self.b, self.gradb)]


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, input):
        if self.training and self.p < 1.:
            self.mask = torch.bernoulli(torch.ones_like(input) - self.p)
            return input * self.mask / (1 - self.p)
        else:
            return input

    def backward(self, gradwrtoutput):
        if self.training and self.p < 1.:
            return gradwrtoutput * self.mask / (1 - self.p)
        else:
            return gradwrtoutput
