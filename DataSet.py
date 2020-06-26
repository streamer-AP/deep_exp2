import torch 
from torch import nn
from glob import glob
import os
import numpy as np
from PIL import Image
from torchvision.transforms import Compose,ToTensor,RandomErasing,RandomResizedCrop,RandomHorizontalFlip,RandomAffine,RandomPerspective,Resize
class DataSet(torch.utils.data.Dataset):
    def __init__(self,img_dir,train=True):
        super().__init__()
        self.train=train
        if self.train:
            cats_path=glob(os.path.join(img_dir,"cat*.jpg"))
            dogs_path=glob(os.path.join(img_dir,"dog*.jpg"))
            self.label=np.zeros((len(cats_path)+len(dogs_path)),dtype=np.int64)
            self.label[:len(cats_path)]=0
            self.label[len(cats_path):]=1
            self.imgs_path=cats_path+dogs_path
        else:
            self.imgs_path=glob(os.path.join(img_dir,"*.jpg"))
    def __getitem__(self,index):
        img=Image.open(self.imgs_path[index])
        if self.train:
            transform=Compose([RandomHorizontalFlip(p=0.5),
            RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3,),
            RandomResizedCrop((414,414),scale=(0.5,1)),
            Resize((414,414)),
            ToTensor(),
            RandomErasing(p=0.3),
            ])
        else:
            transform=Compose([
            Resize((414,414)),
            ToTensor()])
        img=transform(img)
        if self.train:
            label=self.label[index]
            return img,label
        else:
            return img
    
    def __len__(self):
        return len(self.imgs_path)