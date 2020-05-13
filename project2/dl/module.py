import torch


class Module:
    def __init__(self):
        self.training = True

    def forward(self, *input):
        raise NotImplementedError()

    def backward(self, *gradwrtoutput):
        raise NotImplementedError()

    def param(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
