import torch

from .module import Module

# very important
torch.set_grad_enabled(False)


def sigmoid(tensor):
	"""
	Helper function that does the actual sigmoid computation.
	"""
	return 1 / (1 + torch.exp(-tensor))


def tanh(tensor):
	"""
	Helper function that does the actual tanh computation.
	"""
	return 2 / (1+torch.exp(-2*tensor)) - 1


class ReLU(Module):

	def __init__(self):
		super().__init__()

	def name(self):
		return "ReLU"

	def __str__(self):
		return self.name() + "()"

	def forward(self, *input):
		"""
		Applies the forward pass of the ReLU activation.
		Uses Tensor operations to set all negative values to 0.
		"""
		super().forward_check(input)

		# Return ReLU applied to input tensor
		return torch.max(self.input, torch.tensor([0.0]))
	
	def backward(self, *gradwrtoutput):
		"""
		Applies the backward pass of the ReLU activation.
		"""
		super().backward_check(gradwrtoutput)

		# Compute the gradient of ReLU, which is 1 if x > 0 and 0 otherwise
		internal_grad = self.input.clone()
		internal_grad[self.input <= 0] = 0
		internal_grad[self.input > 0] = 1

		return internal_grad * gradwrtoutput[0]


class Sigmoid(Module):

	def __init__(self):
		super().__init__()

	def name(self):
		return "Sigmoid"

	def __str__(self):
		return self.name() + "()"

	def forward(self, *input):
		"""
		Applies the forward pass of the Sigmoid activation.
		"""
		super().forward_check(input)

		return sigmoid(self.input)

	def backward(self, *gradwrtoutput):
		"""
		Applies the backward pass of the Sigmoid activation.
		"""
		super().backward_check(gradwrtoutput)

		# Compute the gradient of Sigmoid
		input_sig = sigmoid(self.input)
		internal_grad = input_sig * (1 - input_sig)
		return internal_grad * gradwrtoutput[0]


class Tanh(Module):

	def __init__(self):
		super().__init__()

	def name(self):
		return "Tanh"

	def __str__(self):
		return self.name() + "()"

	def forward(self, *input):
		"""
		Applies the forward pass of the Tanh activation.
		"""
		super().forward_check(input)

		return tanh(self.input)
	
	def backward(self, *gradwrtoutput):
		"""
		Applies the backward pass of the Tanh activation.
		"""
		super().backward_check(gradwrtoutput)

		internal_grad = 1.0 - tanh(self.input).pow(2)
		return internal_grad * gradwrtoutput[0]


class SELU(Module):
	""" Shifted Exponential Linear Unit activation function.
	SELU activation is self-normalizing if lecun_weight initalization is used"""

	def __init__(self, alpha=1.6732632423, lambda_=1.050700987):
		super().__init__()
		self.alpha = alpha
		self.lambda_ = lambda_
		
	def name(self):
		return "SELU"

	def __str__(self):
		return self.name() + "()"

	def forward(self, *input):
		super().forward_check(input)

		result = self.input.clone()
		result[self.input > 0] = self.lambda_ * self.input[self.input > 0]
		result[self.input <= 0] = self.lambda_ * self.alpha * (torch.exp(self.input[self.input <= 0]) - 1)
		return result
	
	def backward(self, *gradwrtoutput):
		super().backward_check(gradwrtoutput)
		internal_grad = self.input.clone()
		internal_grad[self.input > 0] = self.lambda_
		internal_grad[self.input <= 0] = self.lambda_ * self.alpha * torch.exp(self.input[self.input <= 0])
		return internal_grad * gradwrtoutput[0]
