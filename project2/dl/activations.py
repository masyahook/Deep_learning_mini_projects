import torch
from .module import Module


class ReLU(Module):
    """
        Class ReLU which applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)
        
        Parameters
        ----------
        mask : Torch tensor
            The mask that keeps positive elements of input tensor and zeroes negative elements
    """
    
    def forward(self, input):
        """
            The forward pass of the ReLU class.
            
            Parameters
            ----------
            input : Torch tensor
                The input tensor
            
            Returns
            -------
            Torch tensor
                Input tensor with zeroed negative values
        """
        self.mask = input > 0
        return input * self.mask

    def backward(self, gradwrtoutput):
        """
            The backward pass of the ReLU class (gradient wrt output)
            
            Parameters
            ----------
            gradwrtoutput : Torch tensor
                Tensor of gradients wrt to output
                
            Returns
            -------
            Torch tensor of
                The propagated gradient after ReLU layer
        """
        return gradwrtoutput * self.mask


class Tanh(Module):
    """
        Class Tanh which takes hyperbolic tangent of the input tensor element-wise.
        
        Parameters
        ----------
        output : Torch tensor
            The output tensor
    """
    
    def forward(self, input):
        """
            The forward pass of the Tanh class.
            
            Parameters
            ----------
            input : Torch tensor
                The input tensor
            
            Returns
            -------
            Torch tensor
                The output tensor which is tanh(input) element-wise
        """
        exp = torch.exp(input)
        exp_minus = torch.exp(-input)
        self.output = (exp - exp_minus) / (exp + exp_minus)

        return self.output

    def backward(self, gradwrtoutput):
        """
            The backward pass of the Tanh class (gradient wrt output)
            
            Parameters
            ----------
            gradwrtoutput : Torch tensor
                Tensor of gradients wrt to output
                
            Returns
            -------
            Torch tensor of
                The propagated gradient after Tanh layer
        """
        return (1 - self.output ** 2) * gradwrtoutput


class Sigmoid(Module):
    """
        Class Sigmoid which takes sigmoid function of the input tensor element-wise.
        
        Parameters
        ----------
        output : Torch tensor
            The output tensor
    """
    def forward(self, input):
        """
            The forward pass of the Sigmoid class.
            
            Parameters
            ----------
            input : Torch tensor
                The input tensor
            
            Returns
            -------
            Torch tensor
                The output tensor which is sigmoid(input) element-wise
        """
        self.output = 1 / (1 + torch.exp(-input))

        return self.output

    def backward(self, gradwrtoutput):
        """
            The backward pass of the Sigmoid class (gradient wrt output)
            
            Parameters
            ----------
            gradwrtoutput : Torch tensor
                Tensor of gradients wrt to output
                
            Returns
            -------
            Torch tensor of
                The propagated gradient after Sigmoid layer
        """
        return gradwrtoutput * self.output * (1 - self.output)
