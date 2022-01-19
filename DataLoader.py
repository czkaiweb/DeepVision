import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from matplotlib import image
from torch import Tensor


## Creating a sub class of torch.util.data.Dataset for notMNIST
class notMNIST(Dataset):
	def __init__(self, path):
		Images,Y = [],[]
		folders = os.listdir(path)

		for folder in folders:
			folder_path = os.path.join(path,folder)
			for img in os.listdir(folder_path):
				try:
					img_path = os.path.join(folder_path,img)
					Images.append(np.array(image.imread(img_path)))
					Y.append(ord(folder)-ord("A"))
				except:
					print("File {}/{} is broken".format(folder, img))
		data = [(x, y) for x, y in zip(Images, Y)]
		self.data = data
	
	# The number of items in the dataset
	def __len__(self):
		return len(self.data)

	# getitem is supposed to return (X,Y) for the specific index
	def __getitem__(self, index):
		img = self.data[index][0]

		# 8 bit images. Scale between [0,1]. This helps speed up our training
		img = img.reshape(28, 28) / 255.0

		# Input for Conv2D should be Channels x Height x Width
		img_tensor = Tensor(img).view(1, 28, 28).float()
		label = self.data[index][1]
		return (img_tensor, label)


## Using DataLoader to make sample with sub class notMNIST

