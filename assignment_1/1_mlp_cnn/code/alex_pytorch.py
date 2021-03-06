"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class myAlexNet(nn.Module):
	"""
	This class implements a Convolutional Neural Network in PyTorch.
	It handles the different layers and parameters of the model.
	Once initialized an ConvNet object can perform forward.
	"""
	
	def __init__(self, n_channels, n_classes):
		"""
		Initializes ConvNet object.
		
		Args:
			n_channels: number of input channels
			n_classes: number of classes of the classification problem
			
		
		TODO:
		Implement initialization of the network.
		"""
		
		########################
		# PUT YOUR CODE HERE  #
		#######################
		super(myAlexNet,self).__init__()


		self.net = nn.Sequential(
			nn.Conv2d(3,64, 11,stride=4,padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3,stride =2 ),
			nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, stride=2)
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6,6))
		self.cls = nn.Sequential(
			nn.Dropout(),
			
			nn.Linear(256 * 6 * 6, 3072),
			nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, 10)

		)



		# self.conv0 =		nn.Conv2d(3,64,3, 				stride= 1, padding = 1)
		# self.PreAct1 = 		PreActBlock(64, nn.ReLU, subsample=False, c_out=-1)
		# self.conv1 = 		nn.Conv2d(64,128 ,1, 			stride= 1, padding = 0)
		# self.maxpool1 = 	nn.MaxPool2d(3, 				stride= 2, padding = 1)
		# self.PreAct2_a = 	PreActBlock(128, nn.ReLU, subsample=False, c_out=-1)
		# self.PreAct2_b = 	PreActBlock(128, nn.ReLU, subsample=False, c_out=-1)
		# self.conv2 = 		nn.Conv2d(128,256 ,1, 			stride= 1, padding = 0)
		# self.maxpool2 = 	nn.MaxPool2d(3, 				stride= 2, padding = 1)
		# self.PreAct3_a = 	PreActBlock(256, nn.ReLU, subsample=False, c_out=-1)
		# self.PreAct3_b =	PreActBlock(256, nn.ReLU, subsample=False, c_out=-1)
		# self.conv3 = 		nn.Conv2d(256,512 ,1, 			stride= 1, padding = 0)
		# self.maxpool3 = 	nn.MaxPool2d(3, 				stride= 2, padding = 1)
		# self.PreAct4_a = 	PreActBlock(512, nn.ReLU, subsample=False, c_out=-1)
		# self.PreAct4_b = 	PreActBlock(512, nn.ReLU, subsample=False, c_out=-1)
		# self.maxpool4 = 	nn.MaxPool2d(3, 				stride= 2, padding = 1)
		# self.PreAct5_a = 	PreActBlock(512, nn.ReLU, subsample=False, c_out=-1)
		# self.PreAct5_b = 	PreActBlock(512, nn.ReLU, subsample=False, c_out=-1)
		# self.maxpool5 = 	nn.MaxPool2d(3, 				stride= 2, padding = 1)
		# self.batchN_5 = nn.BatchNorm2d(512)
		# self.Relu5 = nn.ReLU()
		# self.fc = 			nn.Linear(512,10)


		
		print(self)
				
		
		
		
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
		# print(x.shape)
		x = self.net(x)
		x = self.avgpool(x)
		x = torch.flatten(x,1)
		# print(x.shape)
		out = self.cls(x)
		
		# x = self.conv0(x)
		# x = self.PreAct1(x)
		# x = self.conv1(x)
		# x = self.maxpool1(x)
		# x = self.PreAct2_a (x)
		# x = self.PreAct2_b(x)
		# x = self.conv2 (x)
		# x = self.maxpool2(x)
		# x = self.PreAct3_a (x)
		# x = self.PreAct3_b (x)
		# x = self.conv3(x)
		# x = self.maxpool3(x)
		# x = self.PreAct4_a(x)
		# x = self.PreAct4_b(x)
		# x = self.maxpool4(x)
		# x = self.PreAct5_a(x)
		# x = self.PreAct5_b(x)
		# x = self.maxpool5(x)
		# x = self.batchN_5 (x)
		# x = self.Relu5(x)
		# # print(x.shape)
		# x = x.view(x.size(0), -1) 
		
		# # print(x.shape)
		# out =  self.fc(x)
		########################
		# END OF YOUR CODE    #
		#######################
		
		return out
# class PreActBlock(nn.Module):

# 	def __i
# 	nit__(self, c_in, act_fn, subsample=False, c_out=-1):
# 		"""
# 		Inputs:
# 			c_in - Number of input features
# 			act_fn - Activation class constructor (e.g. nn.ReLU)
# 			subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
# 			c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
# 		"""
# 		super().__init__()
# 		if not subsample:
# 			c_out = c_in

# 		# Network representing F
# 		self.net = nn.Sequential(
# 			nn.BatchNorm2d(c_in),
# 			act_fn(),
# 			nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False)

			
# 		)



	# def forward(self, x):
	# 	z = self.net(x)
	# 	out = z + x
	# 	return out
