import torch
from torch.utils.data import Dataset
from augmentations import *
import json
import os
from PIL import Image

class PascalDataset(Dataset):

    def __init__(self,data_folder,ds_type,params):

        
        super(PascalDataset,self).__init__()
        self.ds_type = ds_type.lower()
        self.params = params
        assert self.ds_type in ['train','test']

        self.data_folder = data_folder

        with open(os.path.join(self.data_folder,self.ds_type + '_images.json'),'r') as j:
            self.images = json.load(j)
        with open(os.path.join(self.data_folder,self.ds_type + '_objects.json'),'r') as j:
            self.objects = json.load(j)
        
        assert len(self.objects) == len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):

        img = Image.open(self.images[idx],mode='r')
        img = img.convert('RGB')
        objects = self.objects[idx]
        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels'])
        #difficulties = torch.ByteTensor(objects['difficulties'])

        #if not self.keep_difficult:
        #    boxes = boxes[1 - difficulties]
        #    labels = labels[1 - difficulties]
        #    difficulties = difficulties[1 - difficulties]
        
        img,targets = transformer(img,boxes,labels,self.params)
        
        return img,targets
    
    def collate_fn(self,batch):
        img, targets = [],[]        
            
        for b in batch:
            img.append(b[0])
            targets.append(b[1])

        targets = torch.cat(targets,dim=0)
        img = torch.stack(img,dim = 0)

        return img,targets
