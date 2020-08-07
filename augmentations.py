import torchvision.transforms.functional as TF
import random
import numpy as np
from utils import *
def pad_to_square(img,boxes,pad_value=0,normalized = False):
    """
    input bbox in cx_cy_w_H
    """
    
    w, h = img.size
    w_factor, h_factor = (w,h) if normalized else (1, 1)
    dim_diff = np.abs(h - w)
    pad1= dim_diff // 2
    pad2= dim_diff - pad1

    if h <= w:
        left, top, right, bottom = 0, pad1, 0, pad2
    else:
        left, top, right, bottom = pad1, 0, pad2, 0
    padding = (left, top, right, bottom)
    
    img_padded = TF.pad(img, padding=padding, fill=pad_value)
    w_padded, h_padded = img_padded.size

    x1 = w_factor * boxes[:, 1] 
    y1 = h_factor * boxes[:, 2]
    x2 = w_factor * boxes[:, 3]
    y2 = h_factor * boxes[:, 4]
    
    x1 += padding[0]
    y1 += padding[1]
    

    boxes[:, 1] = (x1) / w_padded
    boxes[:, 2] = (y1) / h_padded
    boxes[:, 3] = x2/ w_padded
    boxes[:, 4] = y2/ h_padded

    return img_padded,boxes


def flip(img,boxes):
    new_img = TF.hflip(img)
    labels = boxes
    labels[:,1] = 1.0 - labels[:,1]
    return new_img, labels

def photometric_distort(img):
    new_img = img
    distortions = [TF.adjust_brightness,
                TF.adjust_contrast,
                TF.adjust_saturation,
                TF.adjust_hue]
    
    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                adjust_factor = random.uniform(-18/255.,18/255.)
            else:
                adjust_factor = random.uniform(0.5,1.5)
            
            new_img = d(new_img,adjust_factor)
    
    return new_img


def transformer(img,targets,labels,params):
    targets = xy_to_cxcy(targets)
    new_label = make_labels(targets,labels)
    if params['pad2square'] is True:
        img,new_label = pad_to_square(img, new_label)
    img = TF.resize(img,params['target_size'])
    if random.random() < params['flip']:
        img,new_label = flip(img,new_label)
    if random.random() < params['photo_distort']:
        img = photometric_distort(img)
    img = TF.to_tensor(img)
    targets = torch.zeros((len(new_label),6))
    targets[:,1:] = new_label

    return img,targets

trans_params_train = {"target_size":(416,416),
                    "pad2square":True,
                    "flip":0.5,
                    "photo_distort":True,
                    }

trans_params_val = {"target_size":(416,416),
                    "pad2square":True,
                    "flip":0.0,
                    "photo_distort":False,
                    }


