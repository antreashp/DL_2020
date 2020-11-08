"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np
from collections import defaultdict

class LinearModule(object):
		"""
		Linear module. Applies a linear transformation to the input data.
		"""
		
		def __init__(self, in_features, out_features):
				"""
				Initializes the parameters of the module.
		
				Args:
					in_features: size of each input sample
					out_features: size of each output sample
		
				TODO:
				Initialize weights self.params['weight'] using normal distribution with mean = 0 and
				std = 0.0001. Initialize biases self.params['bias'] with 0.
		
				Also, initialize gradients with zeros.
				"""
				
				########################
				# PUT YOUR CODE HERE  #
				#######################
				w_shape = (in_features,out_features)
				self.params = defaultdict(float)
				self.grads = defaultdict(float)

				self.params['weights'] = np.random.normal(0,0.0001,w_shape)
				self.params['bias'] = np.zeros(out_features)
				self.grads['weights'] = np.zeros(w_shape)
				self.grads['bias'] = np.zeros(out_features)
				
				
				
				########################
				# END OF YOUR CODE    #
				#######################
		
		def forward(self, x):
				"""
				Forward pass.
		
				Args:
					x: input to the module
				Returns:
					out: output of the module
		
				TODO:
				Implement forward pass of the module.
		
				Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
				"""
				
				########################
				# PUT YOUR CODE HERE  #
				#######################

				self.x = x
				# print(self.params['weights'].shape)
				# print(x.shape)
				# print(x.dot(self.params['weights']).shape)
				self.out = self.x.dot(self.params['weights']) + self.params['bias']
				# self.out = out
				########################
				# END OF YOUR CODE    #
				#######################
				
				return self.out
		
		def backward(self, dout):
				"""
				Backward pass.
		
				Args:
					dout: gradients of the previous module
				Returns:
					dx: gradients with respect to the input of the module
		
				TODO:
				Implement backward pass of the module. Store gradient of the loss with respect to
				layer parameters in self.grads['weight'] and self.grads['bias'].
				"""
				
				########################
				# PUT YOUR CODE HERE  #
				#######################

				self.grads['weights'] = self.x.T.dot(dout)
				self.grads['bias'] = np.sum(dout,axis=0)
				dx = dout.dot(self.params['weights'].T)
				
				########################
				# END OF YOUR CODE    #
				#######################
				return dx



class SoftMaxModule(object):
		"""
		Softmax activation module.
		"""
		
		def forward(self, x):
				"""
				Forward pass.
				Args:
					x: input to the module
				Returns:
					out: output of the module
		
				TODO:
				Implement forward pass of the module.
				To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
		
				Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
				"""
				
				########################
				# PUT YOUR CODE HERE  #
				#######################
				# print('x before softmax:')
				# print(x)
				b = x.max()
				y = np.exp(x-b)
				x = (y.T / y.sum(axis = 1)).T

				self.x = x
				out =x

				# print('x after softmax:')
			
				# print(out)
	# exit()
				# self.x = x
				# b = x.max()
				# y = np.exp(x - b)

				# out =  y / y.sum()
				
				########################
				# END OF YOUR CODE    #
				#######################
				
				return out
		
		def backward(self, dout):
				"""
				Backward pass.
				Args:
					dout: gradients of the previous modul
				Returns:
					dx: gradients with respect to the input of the module
		
				TODO:
				Implement backward pass of the module.
				"""
				
				########################
				# PUT YOUR CODE HERE  #
				#######################
				# print(dout)
				idxs = dout.shape[0]
				dx = np.zeros(dout.shape)
				for i in range(idxs):
					d = self.x[i,:].reshape(-1,1)
					d = np.eye(d.shape[0]).dot(d) - d.dot(d.T)
					dx[i,:] = d.dot(dout[i,:])  
				# print(dx)
				########################
				# END OF YOUR CODE    #
				#######################
				
				return dx


class CrossEntropyModule(object):
	"""
	Cross entropy loss module.
	"""
	
	def forward(self, x, y):
		"""
		Forward pass.
		Args:
			x: input to the module
			y: labels of the input
		Returns:
			out: cross entropy loss

		TODO:
		Implement forward pass of the module.
		"""
		
		########################
		# PUT YOUR CODE HERE  #
		#######################
		
		# print(x)
		# print(y)
		# prediction_idx = x.argmax(1)
		# print(prediction_idx)
		# label_idx = y.argmax(1)
		# print(label_idx)
		# print(x)
		out = -np.log(x[np.arange(x.shape[0]), y.argmax(1)]).mean()
		########################
		# END OF YOUR CODE    #
		#######################
		# print(out)
		return out
	
	def backward(self, x, y):
		"""
		Backward pass.
		Args:
			x: input to the module
			y: labels of the input
		Returns:
			dx: gradient of the loss with the respect to the input x.

		TODO:
		Implement backward pass of the module.
		"""
		
		########################
		# PUT YOUR CODE HERE  #
		#######################
		m  = y.shape[0]
		dx = - (y/x)/m
		
		
		########################
		# END OF YOUR CODE    #
		#######################
		
		return dx


class ELUModule(object):
		"""
		ELU activation module.
		"""
		def __init__(self,alpha=1):
			self.alpha = alpha
		def forward(self, x):
				"""
				Forward pass.

				Args:
					x: input to the module
				Returns:
					out: output of the module

				TODO:
				Implement forward pass of the module.

				Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
				"""
				########################
				# PUT YOUR CODE HERE  #
				#######################
				# print(x)
				idxs = np.where(x>0)
				x[idxs] = self.alpha*(np.exp(x[idxs])-1)
				out = x
				# print(out)
				########################
				# END OF YOUR CODE    #
				#######################
				
				return out
		
		def backward(self, dout):
				"""
				Backward pass.
				Args:
					dout: gradients of the previous module
				Returns:
					dx: gradients with respect to the input of the module

				TODO:
				Implement backward pass of the module.
				"""
				
				########################
				# PUT YOUR CODE HERE  #
				#######################
				# print(dout)
				idxs = np.where(dout>0)
				other_idxs = np.where(dout<0)
				
				dout[idxs] = self.alpha*(np.exp(dout[idxs]))
				dout[other_idxs] = 1
				
				dx = dout
				# print(dx)
				########################
				# END OF YOUR CODE    #
				#######################
				return dx
