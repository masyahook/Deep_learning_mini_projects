import torch
from .module import Module


class Sequential(Module):
    """
        Class Sequential that serves as a container of Modules and acts as a Module itself.
        
        modules : Module object
            The modules of neural network
    """
    def __init__(self, modules):
        """
            Class constructor.
            
            Parameters
            ----------
            modules : Module object
                The modules of neural network
        """
        super().__init__()
        self.modules = modules

    def forward(self, input):
        """
            The forward pass of module container which integrates all forward passes.
            
            Parameters
            ----------
            input : Torch tensor
                The input tensor of neural network
                
            Returns
            -------
            Torch tensor
                The output tensor of neural network
        """
        output = input
        for module in self.modules:
            output = module.forward(output)

        return output

    def backward(self, gradwrtoutput):
        """
            The backward pass of module container which integrates all backward passes.
            
            Parameters
            ----------
            gradwrtoutput : Torch tensor
                Tensor of gradients wrt to output of neural network
                
            Returns
            -------
            Torch tensor
                The accumulated tensor of gradients of input wrt output of neural network
        """
        for module in self.modules[::-1]:
            gradwrtoutput = module.backward(gradwrtoutput)

        return gradwrtoutput

    def param(self):
        """
            Return either a list of lists of tuples with parameters and their gradients for each module
        """
        return [p for module in self.modules for p in module.param()]

    def train(self):
        """
            Setting the module container to training mode (implemented for Dropout layer).
        """
        super().train()

        for module in self.modules:
            module.train()

    def eval(self):
        """
            Setting the module container to evaluation mode (implemented for Dropout layer).
        """
        super().eval()

        for module in self.modules:
            module.eval()
