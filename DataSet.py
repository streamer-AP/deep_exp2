import torch 
from torch import nn
from glob import glob
import os
import numpy as np
from PIL import Image
from torchvision.transforms import Compose,ToTensor
class DataSet(torch.utils.data.DataSet):
    def __init__(self,img_dir,train=True):
        super().__init__()
        self.train=train
        if self.train:
            cats_path=glob(os.path.join(img_dir,"cat*.jpg"))
            dogs_path=glob(os.path.join(img_dir,"dog*.jpg"))
            self.label=np.zeros((len(cats_path)+len(dogs_path),2))
            self.label[:len(cats_path),0]=1
            self.label[len(cats_path):,1]=1
            self.imgs_path=cats_path+dogs_path
        else:
            self.imgs_path=glob(os.path.join(img_dir,"*.jpg"))
    def __getitem__(self,index):
        img=Image.open(self.imgs_path[index])
        transform=Compose([ToTensor])
        img=transform(img)
        if self.train:
            label=self.label[index]
            return label,img
        else:
            return img
    
    def __len__(self):
        return len(self.imgs_path)