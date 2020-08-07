import os
import torch
import random
from torchvision.transforms.functional import to_pil_image
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as TF
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pylab as plt

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k:v+1 for v,k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v:k for k,v in label_map.items()}

fnt = ImageFont.truetype('arial.ttf', 15)
COLORS = np.random.randint(0, 255, size=(80, 3),dtype="uint8")

# color map and dict

def parse_annotation(annot_path):
    tree = ET.parse(annot_path)
    root = tree.getroot()

    boxes, labels, difficulties = [], [], []
    
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1
        
        boxes.append([xmin,ymin,xmax,ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    
    return {'boxes':boxes,'labels':labels,'difficulties':difficulties}


def create_data_lists(voc07_path,voc12_path,output_folder):

    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images, train_objects = [], []
    n_objects = 0

    for path in [voc07_path,voc12_path]:
        
        with  open(os.path.join(path,'Imagesets/Main/trainval.txt','r')) as f:
            ids = f.read().splitlines()

        for id in ids:
            objects = parse_annotation(os.path.join(path,'Annotations',id + '.xml'))
            if len(objects) == 0:
                continue

            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path,'JPEGImages',id + '.jpg'))

        assert len(train_images) == len(train_objects)

        with open(os.path.join(output_folder,'train_images.json'),'w') as j:
            json.dump(train_images,j)
        with open(os.path.join(output_folder,'train_objects.json'),'w') as j:
            json.dump(train_objects,j)
        with open(os.path.join(output_folder,'label_map.json'),'w') as j:
            json.dump(label_map,j)

        print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (len(train_images), n_objects, os.path.abspath(output_folder))) 
            
def plot_img_bbox(img,targets):
    
    if torch.is_tensor(img):
        img = to_pil_image(img)
    
    targ = rescale_bbox(targets,img.size)
    labels = targ[:,0]
    targ = cxcy_to_xy(targ[:,1:])
    if torch.is_tensor(targ):
        targ = targ.numpy()
 
    xmin = targ[:,0]
    ymin = targ[:,1]
    xmax = targ[:,2]
    ymax = targ[:,3]
    
    w,h = img.size
    draw = ImageDraw.Draw(img)
    

    for i in range(targets.shape[0]):
        color = [int(c) for c in COLORS[i]]
        
        draw.rectangle(((xmin[i],ymin[i]),(xmax[i],ymax[i])),
                outline=tuple(color),width = 3)
        
        draw.text((xmin[i],ymin[i]),rev_label_map[int(labels[i])],font = fnt)
    plt.figure(figsize=(8,10))
    plt.imshow(np.array(img))
    plt.axis("off")

def xy_to_cxcy(xy):
    return torch.cat([(xy[:,2:] + xy[:,:2]) / 2, xy[:,2:] - xy[:,:2]], 1)

def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)

def scale_bbox(bbox,size):
    W,H = size
    x = bbox[:,1].unsqueeze(1)/W
    y = bbox[:,2].unsqueeze(1)/H
    w = bbox[:,3].unsqueeze(1)/W
    h = bbox[:,4].unsqueeze(1)/H
    return torch.cat([bbox[:,0].unsqueeze(1),x,y,w,h],dim=1)

def rescale_bbox(bbox, size):
    W,H = size
    x = bbox[:,1].unsqueeze(1)*W
    y = bbox[:,2].unsqueeze(1)*H
    w = bbox[:,3].unsqueeze(1)*W
    h = bbox[:,4].unsqueeze(1)*H
    return torch.cat([bbox[:,0].unsqueeze(1),x,y,w,h],dim=1)

def make_labels(targets,labels):
    assert targets.shape[0] == labels.shape[0]
    lbl = torch.zeros((targets.shape[0],5))
    lbl[:,0] = labels
    lbl[:,1:] = targets
    return lbl
