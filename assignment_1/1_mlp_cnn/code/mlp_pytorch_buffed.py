"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):
	"""
	This class implements a Multi-layer Perceptron in PyTorch.
	It handles the different layers and parameters of the model.
	Once initialized an MLP object can perform forward.
	"""
	
	def __init__(self, n_inputs, n_hidden, n_classes):
		"""
		Initializes MLP object.
		
		Args:
			n_inputs: number of inputs.
			n_hidden: list of ints, specifies the number of units
								in each linear layer. If the list is empty, the MLP
								will not have any linear layers, and the model
								will simply perform a multinomial logistic regression.
			n_classes: number of classes of the classification problem.
									This number is required in order to specify the
									output dimensions of the MLP

		TODO:
		Implement initialization of the network.
		"""
		
		########################
		# PUT YOUR CODE HERE  #
		#######################
		super(MLP,self).__init__()
		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		print(n_hidden)
		self.layers = nn.ModuleList()
		# self.layers = []
		self.layers.append(nn.Linear(n_inputs,n_hidden[0])) # inpt layer
		# self.layers.append(nn.Elu())
		# print(n_hidden[1])
		if len(n_hidden)>1:

			for n_h in range(1,len(n_hidden)):
				# print(n_h)
				# print(n_h)
				# exit()
				self.layers.append(nn.Linear(n_hidden[n_h-1],n_hidden[n_h]))
			# self.layers.append(nn.Elu())
			# for i in self.n_hidden:
		# for l in range(len(self.layers)):
			# nn.init.xavier_uniform(self.layers[l].weight)
			# self.layers[l].bias.data.fill_(0.01)
# 
			
		self.layers.append(nn.Linear(n_hidden[-1],n_classes))
		# nn.init.xavier_uniform(self.layers[-1].weight)
		# self.layers[-1].bias.data.fill_(0.01)
		self.drop = nn.Dropout(p=0.2)
		self.Relu = nn.ReLU()
		self.Elu = nn.ELU()
		self.soft = nn.Softmax(dim=1)
		# self.layers.append(nn.Softmax(dim=1))
		print(self.layers)
			

		########################
		# END OF YOUR CODE    #
		#######################
	
	def forward(self, x):
		"""
		Performs forward pass of the input. Here an input tensor x is transformed through
		several layer transformations.
		
		Args:
			x: input to the network
		Returns:
			out: outputs of the network
		
		TODO:
		Implement forward pass of the network.
		"""
		
		########################
		# PUT YOUR CODE HERE  #
		#######################
		for l in self.layers[:-1]:
			
			x = l(x)
			x = self.drop(x)
			x = self.Elu(x)
			
		# x = self.drop(x)
		out = self.layers[-1](x)
		
		# out = self.soft(x)
		########################
		# END OF YOUR CODE    #
		#######################
		
		return out
