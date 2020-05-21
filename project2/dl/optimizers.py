class Optimizer:
    """
        Parent class for all implemented optimizers.

        Parameters
        ----------
        param : tuple of two torch tensors
            Pair of parameter tensor and a gradient tensor
    """
    
    def __init__(self, param):
        """
            Class constructor.
        """
        self.param = param

    def step(self):
        """
            Optimizer step for updating parameter values.
        """
        raise NotImplementedError()

    def zero_grad(self):
        """
            Method that resets parameter gradients.
        """
        for _, g in self.param:
            g[:] = 0.


class SGD(Optimizer):
    """
        Class for stochastic gradient descent optimizer.
        
        Parameters
        ----------
        param : tuple of two torch tensors
            Pair of parameter tensor and a gradient tensor
        lr : float
            Learning rate of the SGD optimizer
    """
    
    def __init__(self, param, lr=1.):
        """
            Class constructor.
            
            Parameters
            ----------
            param : tuple of two torch tensors
                Pair of parameter tensor and a gradient tensor
            lr : float
                Learning rate of the SGD optimizer (by default 1)
        """
        super().__init__(param)

        self.lr = lr


    def step(self):
        """
            SGD optimizer step that updates the network parameters in the direction of
            negative gradient evaluated on subset of the data.
            
        """
        for p, grad in self.param:
            p[:] = p - self.lr * grad
