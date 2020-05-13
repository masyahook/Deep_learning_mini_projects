class Optimizer:
    def __init__(self, param):
        self.param = param

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for _, g in self.param:
            g[:] = 0.


class SGD(Optimizer):
    def __init__(self, param, lr=1.):
        super().__init__(param)

        self.lr = lr


    def step(self):
        for p, grad in self.param:
            p[:] = p - self.lr * grad
