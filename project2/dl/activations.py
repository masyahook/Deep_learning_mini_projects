import torch
from .module import Module


class ReLU(Module):
    def forward(self, input):
        self.mask = input > 0
        return input * self.mask

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.mask


class Tanh(Module):
    def forward(self, input):
        exp = torch.exp(input)
        exp_minus = torch.exp(-input)
        self.output = (exp - exp_minus) / (exp + exp_minus)

        return self.output

    def backward(self, gradwrtoutput):
        return (1 - self.output ** 2) * gradwrtoutput


class Sigmoid(Module):
    def forward(self, input):
        self.output = 1 / (1 + torch.exp(-input))

        return self.output

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.output * (1 - self.output)
