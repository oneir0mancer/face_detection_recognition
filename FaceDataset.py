import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *

class FaceDataset(Dataset):
    def __init__(self, metadata, subjects, size = (5,5), anchors=[(60,60), (160,160), (240,240)], input_size=(640, 640)):
        
        self.metadata = metadata
        self.subjects = subjects
        self.n_classes = len(subjects)
        self.anchors = anchors
        self.feature_map_size = size
                           
        #print('{} images'.format(len(self.images)))

    def __len__(self):
        return len(self.metadata)
               
    def __getitem__(self, idx):
        #Load image
        img = Image.open(metadata[idx]['path'])
        subj =  metadata[idx]['subject']
        bbox = metadata[idx]['rect']    #(top, bottom, left, right)
        
        #Crop         
        crop_rect = CleverRandomCropArea(bbox, img.size, crop_size=input_size)
        img = img.crop(crop_rect)
        bbox = Crop(bbox, crop_rect)
        
        top, bottom, left, right = bbox
        # Get gt boxes
        # x,y,h,w,class_label
        gt_boxes = np.zeros((1, 4+1), dtype=np.float32)

        #Relative center and size
        center_x = (left+right)/2/img.width
        center_y = (top+bottom)/2/img.height
        bbox_height = abs(top - bottom)/img.width
        bbox_width = abs(right - left)/img.height
        subj_index = self.subjects.index(subj['subject'])
        
        gt_boxes[0, :] = center_x, center_y, bbox_height, bbox_width, subj_index
        
        #ToTensor
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_boxes = transforms.ToTensor()(gt_boxes)
        
        return {'img': img, 'target': gt_boxes}
