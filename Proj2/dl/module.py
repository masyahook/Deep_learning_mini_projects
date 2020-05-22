import torch


class Module:
    """
        Base class for all implemented neural network modules.
        
        Parameters
        ----------
        training : boolean
            When True the module is set to train mode, when False to test mode.
            This parameter is needed when using Dropout for training and inferring.
    """
    def __init__(self):
        """
            Initiation of the class.
        """
        self.training = True

    def forward(self, *input):
        """
            Forward pass of the module.
        """
        raise NotImplementedError()

    def backward(self, *gradwrtoutput):
        """
             Backward pass of the module (gradient wrt output).
        """
        raise NotImplementedError()

    def param(self):
        """
            Return either a list of tuples with parameter tensor and gradient tensor, or empty list for parameterless modules.
        """
        return []

    def train(self):
        """
            Setting the module to training mode (implemented for Dropout layer).
        """
        self.training = True

    def eval(self):
        """
            Setting the module to evaluation mode (implemented for Dropout layer).
        """
        self.training = False
