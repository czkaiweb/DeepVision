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
		return x