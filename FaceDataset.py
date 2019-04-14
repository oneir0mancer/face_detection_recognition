import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from box_transforms import *

class FaceDataset(Dataset):
    def __init__(self, metadata, box_transform=None, img_transform=None):
        
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)
               
    def __getitem__(self, idx):
        #Load image
        img = Image.open(self.metadata[idx]['path'])
        
        np_im = np.array(img)[:,:,:3]    #delete alpha if png
        img = Image.fromarray(np_im)
        
        subj = self.metadata[idx]['subject']
        bbox = self.metadata[idx]['rect']    #(top, bottom, left, right)
        
        if box_transform:
            img, bbox = box_transform(img, bbox)        
        
        top, bottom, left, right = bbox
        # Get gt boxes
        # x,y,h,w,class_label
        gt_boxes = np.zeros((1, 4+1), dtype=np.float32)

        #Relative center and size
        center_x = (left+right)/2/img.width
        center_y = (top+bottom)/2/img.height
        bbox_height = abs(top - bottom)/img.width
        bbox_width = abs(right - left)/img.height
        subj_index = subj
        
        gt_boxes[0, :] = center_x, center_y, bbox_height, bbox_width, subj_index
        
        #ToTensor
        if img_transform:
            img = img_transform(img)
        
        return {'img': img, 'target': gt_boxes}
