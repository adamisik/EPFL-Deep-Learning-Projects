import torch
import math

from .module import Module

# very important
torch.set_grad_enabled(False)


class Linear(Module):

    def __init__(self, in_dim, out_dim, initialization="standard"):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create parameters
        self.weights = torch.empty((out_dim, in_dim))
        self.bias = torch.empty(out_dim)

        self.weights_grad = torch.empty((out_dim, in_dim))
        self.bias_grad = torch.empty(out_dim)

        # Initialize parameters
        self.zero_grad()

        if initialization == "standard":
            limit = 1 / math.sqrt(in_dim)
            self.weights = self.weights.uniform_(-limit, limit)
            self.bias = self.bias.uniform_(-limit, limit)
        elif initialization == "normal":
            self.weights = self.weights.normal_()
            self.bias = self.bias.normal_()
        elif initialization == "zeros":
            self.weights = self.weights.zero_()
            self.bias = self.bias.zero_()
        elif initialization == "ones":
            self.weights = self.weights.fill_(1)
            self.bias = self.bias.fill_(1)
        else:
            raise ValueError("Unknown initialization parameter")

    def name(self):
        return "Linear"

    def __str__(self):
        return self.name() + "(" + str(self.in_dim) + ", " + str(self.out_dim) + ")"

    def forward(self, *input):
        super().forward_check(input)
        return self.input @ self.weights.T + self.bias

    def backward(self, *gradwrtoutput):
        super().backward_check(gradwrtoutput)
        self.weights_grad += gradwrtoutput[0].T @ self.input
        self.bias_grad += gradwrtoutput[0].sum(dim=0)
        return gradwrtoutput[0] @ self.weights

    def params(self):
        return [self.weights, self.bias]

    def params_with_grads(self):
        return [
            (self.weights, self.weights_grad),
            (self.bias, self.bias_grad)
        ]

    def zero_grad(self):
        self.weights_grad.zero_()
        self.bias_grad.zero_()


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def name(self):
        return "Sequential"

    def __str__(self):
        string = self.name() + "(\n"
        for module in self.modules:
            string += "    " + module.__str__() + "\n"
        string += ")"
        return string

    def forward(self, *input):
        """
        Applies the forward pass of the Sequential module.
        """
        super().forward_check(input)
        # Iterate on the internal modules from left to right and apply their .forward()
        neurons = self.input
        for module in self.modules:
            neurons = module.forward(neurons)
        return neurons

    def backward(self, *gradwrtoutput):
        """
        Applies the forward pass of the Sequential module.
        """
        super().backward_check(gradwrtoutput)
        # Iterate on the internal modules from right to left and apply their .backward()
        grad = gradwrtoutput[0]
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def params(self):
        return [param for module in self.modules for param in module.params()]

    def params_with_grads(self):
        return [(p, g) for module in self.modules for (p, g) in module.params_with_grads()]

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
