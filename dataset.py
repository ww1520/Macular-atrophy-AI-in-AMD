
import os
import cv2
import torchvision
from PIL import Image

from torch.utils.data import Dataset
from torchvision.utils import save_image

import torch
import torch.nn as nn
from torch.nn import functional as F

IMG_PATH= r'C:\\Users\\lr3118\\Desktop\\data\\JPEGImages'
LABEL_PATH=r'C:\Users\lr3118\Desktop\data\labels'


class SEGData(Dataset):
    def __init__(self):
        
        self.img_path=IMG_PATH
        self.label_path=LABEL_PATH
        self.label_data=os.listdir(self.label_path)
        self.totensor=torchvision.transforms.ToTensor()
        self.resizer=torchvision.transforms.Resize((256,256))
        
    def __len__(self):
        return len(self.label_data)
    def __getitem__(self, item):
        
        img_name = os.path.join(self.label_path, self.label_data[item])
        img_name = os.path.split(img_name)
        img_name = img_name[-1]
        img_name = img_name.split('.')
        img_name = img_name[0] + '.png'
        img_data = os.path.join(self.img_path, img_name)
        label_data = os.path.join(self.label_path, self.label_data[item])
        
        img = Image.open(img_data)
        label = Image.open(label_data)
        w, h = img.size
        
        slide = max(h, w)
        black_img = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_label = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_img.paste(img, (0, 0, int(w), int(h)))  
        black_label.paste(label, (0, 0, int(w), int(h)))
        
        img = self.resizer(black_img)
        label = self.resizer(black_label)
        img = self.totensor(img)
        label = self.totensor(label)
        return img,label