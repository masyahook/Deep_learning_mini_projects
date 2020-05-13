import torch
from .module import Module


class Sequential(Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, input):
        output = input
        for module in self.modules:
            output = module.forward(output)

        return output

    def backward(self, gradwrtoutput):
        for module in self.modules[::-1]:
            gradwrtoutput = module.backward(gradwrtoutput)

        return gradwrtoutput

    def param(self):
        return [p for module in self.modules for p in module.param()]

    def train(self):
        super().train()

        for module in self.modules:
            module.train()

    def eval(self):
        super().eval()

        for module in self.modules:
            module.eval()
