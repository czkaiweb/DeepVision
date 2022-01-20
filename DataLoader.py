import os,sys
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


# Dataset to use CIFAR database
import pickle

class CIFAR(Dataset):
    def __init__(self,path): 
        self.dataPath = path
        if isinstance(self.dataPath,list):
            batches = list(map(self.unpickle,self.dataPath))
            self.data = np.concatenate([batch["data"] for batch in batches]).reshape([-1,3,32,32]).astype('float32')/255
            self.label = np.concatenate([batch["labels"] for batch in batches]).astype('int32')
        elif isinstance(self.dataPath,str):
            batch = self.unpickle(self.dataPath)
            self.data = batch["data"].reshape([-1,3,32,32]).astype('float32')/255
            self.label = np.array(batch["labels"]).astype('int32')
        else:
            print("Invalid path")
        
        
    def unpickle(self,file):
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            dict = pickle.load(fo)
        else:
            dict = pickle.load(fo,encoding='latin1')
    
        fo.close()
        return dict
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        img = self.data[index]
        # 8 bit images. Scale between [0,1]. This helps speed up our training
        # Input for Conv2D should be Channels x Height x Width
        img_tensor = Tensor(img).view(3, 32, 32).float()
        label = self.label[index]
        return (img_tensor, label)



