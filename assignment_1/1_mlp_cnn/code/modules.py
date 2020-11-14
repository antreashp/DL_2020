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
				# print(self.out)
				# self.out
				print('output of linear f: ',self.out)
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
				# print(self.x.shape)
				# print(dout.shape)

				self.grads['weights'] = self.x.T.dot(dout)
				self.grads['bias'] =  np.sum(dout,axis=0,keepdims=True)
				dx = dout.dot(self.params['weights'].T)
				# print(dx.shape)
				# print(self.grads['weights'].shape)
				# exit()
				########################
				# END OF YOUR CODE    #
				#######################
				print('output of linear b: ',dx)
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
				print(x)
				s = np.max(x, axis=1)
				s = s[:, np.newaxis] # necessary step to do broadcasting
				e_x = np.exp(x - s)
				div = np.sum(e_x, axis=1)
				div = div[:, np.newaxis] 
				self.x = e_x / div
				print('sanity of out of softmax f: ',self.x.sum(1))
				# print(self.x.sum(1))
				return self.x

				print('x before softmax:')
				# print(x)
				print(x)
				b = np.max(x)
				y = np.exp(x-b)
				# print(y)
				# print(y)
				# exit()
				self.x = (y / y.sum(axis = 1))

				# self.x = x
				out =self.x

				print('x after softmax:')

				print(out)
				print('out sum:')

				print(out.sum(1))
	# exit()	
				exit()
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
				
				ds_x = self.x[:,:,None] *self.x[:,None,:]

				for i in range(len(self.x)):
					ds_x[i,:,: ] += np.diag(self.x[i,:])
				dx = np.einsum('ij, ijk -> ik',dout,ds_x)

				print('sanity of out of softmax b: ',dx)
				return dx
				# J = -np.outer(self.x,self.x) + np.diag(self.x.flatten())
				# return J
				# idxs = dout.shape[0]
				# dx = np.zeros(dout.shape)
				# for i in range(idxs):
				# 	d = self.x[i,:].reshape(-1,1)
				# 	d = np.eye(d.shape[0]).dot(d) - d.dot(d.T)
				# 	dx[i,:] = d.dot(dout[i,:])  
				# # print(dx.sum(0))
				# ########################
				# # END OF YOUR CODE    #
				# #######################
				
				# return dx


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
		# print(x)
		# x = x.clip(min=1e-8,max=None)
		# out =  (np.where(y==1,-np.log(x), 0)).sum(axis=1).mean()
		# for i in range(x.shape[0]):
		# 	x[i]  
		# print(x)
		if x.any() <0:
			print('here')
		# print(x.sum(1))
		# if x.sum() != 200:
		# 	print('meeh')
		out = - np.array( y*np.log(x)).sum(axis=1).mean()
		# out = -np.log(x[np.arange(x.shape[0]), y.argmax()]).mean()
		# exit()
		########################
		# END OF YOUR CODE    #
		#######################

		print('output of CrossE f: ',out)
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
		# print(x.shape)
		# print(y.shape)

		dx = (- y / x)/len(y)

		# y = y.argmax(axis=1)
		# m  = y.shape[0]
		# x[range(m),y] -= 1
		# dx = x/m

		
		# print(y.shape)
		# for i in range(m):
		# 	dx = -(1/x[i]) if y[i]== 1 else 0
		# dx = - (y/x)/m
		
		
		########################
		# END OF YOUR CODE    #
		#######################
		
		print('output of CrossE b: ',dx)
		return dx


class ELUModule(object):
		"""
		ELU activation module.
		"""
		def __init__(self,alpha=1):
			self.alpha = alpha
			# self.grads = defaultdict(float)
			# self.grads['weights'] = np.zeros(w_shape)
			# self.grads['bias'] = np.zeros(out_features)
		def single_forward(self,z):
			out = z if z >= 0 else self.alpha*(np.exp(z) -1)
			return out
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
				# self.idxs = (x>0)
				# print(x)
				# print(self.idxs)
				# print(self.idxs)
				# out = []
				# print(x.shape)
				# print(type(x))
				# print(x.shape)
				out = np.vectorize(self.single_forward)(x)#map(self.single_forward,x)
				# print(out.shape)
				self.z = x
				# self.idxs = []
				# for i in range(x.shape[0]):
				# 	out.append([])
				# 	self.idxs.append([])
				# 	for j in range(x.shape[1]):
				# 		if x[i][j] <= 0:
				# 			self.idxs[i].append(True)
				# 			out[i].append(self.alpha*(np.exp(x[i][j])-1))
				# 		else:
				# 			# print(x[i][j])
				# 			self.idxs[i].append(False)
				# 			out[i].append(x[i][j])
							# print('here')
							# exit()
				# ou

				out = np.array(out)
				# print(out.shape)
				# print(out)
				# exit()
				# out = x
				
				# print(out)
				########################
				# END OF YOUR CODE    #
				#######################
				
				print('output of ELU f: ',out)
				return out
		def single_backward(self,z):
			dx = z if z >= 0 else self.alpha*(np.exp(z)-1)
			return dx
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

				# idxs = np.where(dout>0)
				# print(self.idxs)
				# print(self.idxs)
				dx = np.vectorize(self.single_backward)(self.z)
				# print(dx.shape)
				# print(self.idxs.shape)
				# print(self.idxs)

				# dx = 1 if z > 0 else self.alpha*np.exp(z)
				# for k,(i,j) in enumerate(zip(self.idxs,dout)):
				# 	for l,(ii,jj) in enumerate(zip(i,j)):
				# 		# print(ii,jj)
				# 		if ii:
				# 			dout[k,l] = self.alpha*(np.exp(dout[k,l]))
				# 		else:
				# 			dout[k,l] = 1
					
				# exit()
				# # other_idxs = not self.idxs.all()
				# # for i in range(dout.shape[0]):
				# # 	for j in range(dout.shape[1]):
				# # 		if i == 
				# dout[self.idxs] = self.alpha*(np.exp(dout[self.idxs]))
				# dout[other_idxs] = 1
				
				# dx = dout
				# print(dx.shape)
				# print(dx)
				########################
				# END OF YOUR CODE    #
				#######################

				print('output of ELU b: ',dx)
				return dx
