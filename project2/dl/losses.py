import torch
from .module import Module


class LossMSE(Module):
    """
        Class for MSE loss - Mean Squared Error.
        
        Parameters
        ----------
        diff : Torch tensor of size (N, *)
            Difference between input and target tensors 
    """
    
    def forward(self, input, target):
        """
            The forward pass of the MSE loss (direct calculation).
            
            Parameters
            ----------
            input : Torch tensor of size (N, *)
                Input tensor 
            target : Torch tensor of size (N, *)
                Target tensor 
            
            Returns
            -------
            Torch tensor of size (N, *)
                The calculated MSE loss between input and target          
        """
        
        self.diff = input - target
        return (self.diff ** 2).sum() / input.shape[0]

    def backward(self):
        """
            The backward pass of the MSE loss (gradient).
            
            Returns
            -------
            Torch tensor of size (N, *)
                The MSE loss gradient
        """
        return 2 * self.diff / self.diff.shape[0]


class LossBCE(Module):
    """
        Class for BCE loss - Binary Cross Entropy.
        
        Parameters
        ----------
        EPS : float
            Small number that we introduce for numerical stability of calculations.
        input : Torch tensor of size (N, *)
            Input tensor 
        target : Torch tensor of size (N, *)
            Target tensor 
    """
    
    def __init__(self, eps=1e-7):
        """
            Class constructor.
            
            Parameters
            ----------
            eps : float
                Small number that we introduce for numerical stability of calculations.
        """
        super().__init__()
        self.EPS = eps

    def forward(self, input, target):
        """
            The forward pass that clamps the input value and outputs the BCE loss.
            
            Returns
            -------
            Torch tensor of size (N, *)
                The BCE loss between input and target
        """
        self.input = torch.clamp(input, self.EPS, 1 - self.EPS)
        self.target = target

        return (-target * self.input.log() - (1 - target) * (1 - self.input).log()).sum() / self.input.shape[0]

    def backward(self):
        """
            The backward pass of the BCE loss.
            
            Returns
            -------
            Torch tensor of size (N, *)
                The BCE loss gradient
        """
        return (-self.target / self.input + (1 - self.target) / (1 - self.input)) / self.input.shape[0]
