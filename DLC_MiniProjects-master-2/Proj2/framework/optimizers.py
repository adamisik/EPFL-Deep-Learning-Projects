import torch

# very important
torch.set_grad_enabled(False)


class SGD:
    """
     SGD Optimizer with configurable learning rates, moment

     """
    def __init__(self, module):
        self.module = module

    def zero_grad(self):
        """Reset current param. gradients

         Arguments:
             self {[bias_grad]} -- [Bias gradient]
             self {[weight_grad]} -- [Weight gradient]

         Returns:
             [bias_grad, weight_grad] -- [Gradients set to 0]
         """
        self.module.zero_grad()

    def step(self, lr):
        """Stochastic gradient step with learning rate lr.

        Arguments:
            lr {[lr]} -- [Learning rate eta]
        """
        for (p, g) in self.module.params_with_grads():
            p -= lr * g
