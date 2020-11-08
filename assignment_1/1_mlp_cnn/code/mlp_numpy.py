"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
		"""
		This class implements a Multi-layer Perceptron in NumPy.
		It handles the different layers and parameters of the model.
		Once initialized an MLP object can perform forward and backward.
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

				# self.n_inputs = n_inputs
				# self.n_hidden = n_hidden
				# self.n_classes = n_classes
				# self.input_layer = [Neuron() for x in self.n_inputs]
				# for h in self.n_hidden:
				# 	self.hidden_layers.append( [Neuron() for x in h])
				# self.output_layer = [Neuron(activation='out') for x in self.n_classes]
				


				########################
				# PUT YOUR CODE HERE  #
				#######################
				self.layers = []
				self.layers.append(LinearModule(n_inputs,n_hidden[0])) # inpt layer
				for n_h in range(len(n_hidden[1:])):
					self.layers.append(LinearModule(n_hidden[n_h-1],n_h))

				self.out_layer = LinearModule(n_hidden[-1],n_classes)
				self.Elu = ELUModule(alpha = 1)
				self.soft = SoftMaxModule() 
				# print(self.layers[0])
				# print(self.out_layer)
				# exit()
				########################
				# END OF YOUR CODE    #
				#######################

		def forward(self, x):
				# self.input_layer
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
				# print(x[0])
				# print(x[0].shape)
				# norm = [float(i)/sum(x[0]) for i in x[0]]
				# norm = [(float(i)-min(x[0]))/(max(x[0])-min(x[0])) for i in x[0]]
				# print(norm)

				########################
				# PUT YOUR CODE HERE  #
				#######################
				for l in self.layers: # input and hidden 
					x = l.forward(x)
					x = self.Elu.forward(x)
				x = self.out_layer.forward(x)
				out = self.soft.forward(x)

				########################
				# END OF YOUR CODE    #
				#######################

				return out

		def backward(self, dout):
				"""
				Performs backward pass given the gradients of the loss.

				Args:
					dout: gradients of the loss

				TODO:
				Implement backward pass of the network.
				"""

				########################
				# PUT YOUR CODE HERE  #
				#######################
				dout = self.soft.backward(dout)
				dout = self.out_layer.backward(dout)
				for l in self.layers[::-1]:
					dout = self.Elu.backward(dout)
					dout = l.backward(dout)
				########################
				# END OF YOUR CODE    #
				#######################

				return dout
		# def make_layer():



# class Neuron(object):
# 	def __init__(self, activation=None,weight=None )
# 		if weight is None:
# 			self.weight = 0
# 		else:
# 			self.weight = weight
		
# 		if activation is None:
# 			self.activation = 'Relu'
# 		else:
# 			self.activation = activation
	