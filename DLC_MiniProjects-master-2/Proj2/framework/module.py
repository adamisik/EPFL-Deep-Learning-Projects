import torch

# very important
torch.set_grad_enabled(False)


class Module(object):

	def __init__(self):
		self.input = None

	def name(self):
		return "Module"

	def forward(self, *input):
		raise NotImplementedError
	
	def backward(self, *gradwrtoutput):
		raise NotImplementedError
	
	def params(self):
		return []

	def params_with_grads(self):
		return []

	def zero_grad(self):
		pass

	def forward_check(self, input):
		"""
		Intermediate method that verifies size of input and saves it in a variable for later use.
		To be called in the beginning of every Modules's forward method if it takes only one argument.
		"""
		if len(input) > 1:
			raise ValueError(self.name() + ".forward expects a single input argument!")

		# Save input for later
		self.input = input[0].clone()

	def backward_check(self, gradwrtoutput):
		"""
		Intermediate method that verifies size of gradwrtoutput and checks if forward was called previously.
		To be called in the beginning of every Modules's backward method if it takes only one argument.
		"""
		if len(gradwrtoutput) > 1:
			raise ValueError(self.name() + ".backward expects a single input argument!")

		if self.input is None:
			raise ValueError(self.name() + ".backward called before" + self.name() + ".forward!")
