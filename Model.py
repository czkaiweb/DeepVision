from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.lin1 = nn.Linear(784,128)
		self.lin2 = nn.Linear(128,10)



	def forward(self, x):
		x = F.sigmoid(self.lin1(x))
		x = F.softmax(self.lin2(x))

		# Reshaping the tensor to BATCH_SIZE x 320. Torch infers this from other dimensions when one of the parameter is -1.
		x = x.view(-1, 10)
		return x


class CIFAR_NN(nn.Module):
	def __init__(self):
		super(CIFAR_NN,self).__init__()
		self.flatten = nn.Flatten()
		self.dense1 = nn.Linear(3*32*32,64)
		self.logits10 = nn.Linear(64,10)

	def forward(self,x):
		x = F.relu(self.flatten(x))
		x = self.logits10(self.dense1(x))
		#x = F.softmax(x)
		return x

class CIFAR_CNN(nn.Module):
	def __init__(self, useBatchNorm = False):
		super(CIFAR_CNN,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3)
		self.batchNorm1 = nn.BatchNorm2d(10)
		self.pool1 = nn.MaxPool2d(2)
		self.flat1 = nn.Flatten()
		self.dens1 = nn.Linear(15*15*10,100)
		self.batchNorm2 = nn.BatchNorm1d(100)
		self.drop  = nn.Dropout(0.1)
		self.dens2 = nn.Linear(100,10)


	def forward(self,x):
		x = self.conv1(x)
		if useBatchNorm:
			x = self.batchNorm1(x)
		x = F.relu(x)
		x = self.pool1(x)
		x = self.flat1(x)
		x = self.dens1(x)
		if useBatchNorm:
			x = self.batchNorm2(x)
		x = F.relu(x)
		x = self.drop(x)
		x = self.dens2(x)
		#x = F.softmax(x)
		return x