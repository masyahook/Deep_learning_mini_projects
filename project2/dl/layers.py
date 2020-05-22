import torch
from .module import Module


class Linear(Module):
    """
        Class for Linear Module that applies linear transformation to the input data: y = x*W^T + b.
        
        Parameters
        ----------
        W : Torch tensor of size (n_out, n_in)
            The weight matrix parameters
        b : Torch tensor of size n_out
            The bias weight vector of parameters
        gradW : Torch tensor of size (n_out, n_in)
            The gradient wrt to matrix parameters
        gradb : Torch tensor of size n_out
            The gradient wrt to bias parameters
        input : Torch tensor of size (N, n_in)
            Input tensor 
    """
    
    def __init__(self, n_in, n_out):
        """
            Class constructor.
            
            Parameters
            ----------
            n_in : int
                size of each input sample
            n_out : int
                size of each output sample
        """
        super().__init__()

        # Smart weight initialization 
        stdv = 1. / n_in ** 0.5
        self.W = (torch.rand(n_out, n_in) - 0.5) * 2 * stdv
        self.b = (torch.rand(n_out) - 0.5) * 2 * stdv
        
        # Setting gradients wrt to parameters to zero
        self.gradW = torch.zeros_like(self.W)
        self.gradb = torch.zeros_like(self.b)

    def forward(self, input):
        """
            The forward pass of the linear class - calculates linear transformation of the data.
            
            Parameters
            ----------
            input : Torch tensor of size (N, n_in)
                Input tensor
                
            Returns
            -------
            Torch tensor of size (N, n_out)
                The output of the linear layer
        """
        self.input = input

        return input @ self.W.T + self.b

    def backward(self, gradwrtoutput):
        """
            The backward pass of the linear class (gradient wrt output).
            
            Parameters
            ----------
            gradwrtoutput : Torch tensor of size (N, n_out)
                Tensor of gradients wrt to output
                
            Returns
            -------
            Torch tensor of size (N, n_in)
                The propagated gradient after linear layer
        """
        
        # Accumulating gradient wrt to parameters across N samples
        self.gradW += gradwrtoutput.T @ self.input
        self.gradb += gradwrtoutput.sum(axis=0)

        return gradwrtoutput @ self.W

    def param(self):
        """
            Return either a list of tuples with parameters and their gradients
        """
        return [(self.W, self.gradW), (self.b, self.gradb)]


class Dropout(Module):
    """
        Class for Dropout Module that during training randomly zeroes some of the elements of the 
        input tensor with probability p using samples from a Bernoulli distribution
        
        Parameters
        ----------
        p : float
            Probability of a certain element to be zeroed
        mask : Torch tensor
            Mask that keeps or zeroes the elements
    """
    
    def __init__(self, p=0.5):
        """
            Class constructor.
            
            Parameters
            ----------
            p : float
                Probability of a certain element to be zeroed
        """
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, input):
        """
            The forward pass of the dropout layer that zeroes random elements in the input tensor.
            
            Parameters
            ----------
            input : Torch tensor
                Input tensor
                
            Returns
            -------
            Torch tensor
                Either randomly zeroed input tensor or untouched input tensor
        """
        # If model is in training mode the zero random elements
        if self.training and self.p < 1.:
            self.mask = torch.bernoulli(torch.ones_like(input) - self.p)
            return input * self.mask / (1 - self.p)
        else:
            return input

    def backward(self, gradwrtoutput):
        """
            The backward pass of the dropout layer (gradient wrt output).
            
            Parameters
            ----------
            gradwrtoutput : Torch tensor
                Tensor of gradients wrt to output
                
            Returns
            -------
            Torch tensor of
                The propagated gradient after dropout layer
        """
        if self.training and self.p < 1.:
            return gradwrtoutput * self.mask / (1 - self.p)
        else:
            return gradwrtoutput
