import torch

from .module import Module

# very important
torch.set_grad_enabled(False)


class LossMSE(Module):
	"""Mean squared loss with l2 norm
	"""
	def __init__(self):
		super().__init__()
		self.target = None

	def name(self):
		return "LossMSE"
	
	def forward(self, input, target):
		"""
		Parameters
		----------
		input : tensor
			values computed by the network
		target : tensor
			expected outcome

		Returns
		-------
		tensor
			mean squared error
		"""
		self.input = input
		self.target = target

		return (input-target).pow(2).mean()

	def backward(self):
		if self.input is None and self.target is None:
			raise ValueError(self.name() + ".backard called before " + self.name() + ".forward!")
		return 2 * (self.input-self.target)/self.input.shape[1]
