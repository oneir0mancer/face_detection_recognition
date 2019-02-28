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
        
        #Channels: x,y,w,h + confidence + class distribution 
        n_ancors = len(self.anchors)
        feature_map = np.zeros((n_ancors*(4+1+n_classes), feature_map_size[0], feature_map_size[1]), dtype=np.float32)
        
        #Construct feature_map from metadata
        subj_index = self.subjects.index(subj['subject'])
        
        top, bottom, left, right = bbox
        
        #Relative center of bbox
        center_x = (left+right)/2/img.width
        center_y = (left+right)/2/img.height

        #Get cell index and relative offset
        cells = np.linspace(0, 1, self.feature_map_size[0] + 1)
        cell_index_x = np.argmax(cells > center_x)-1
        offset_x = (center_x - cells[cell_index_x])/(cells[cell_index_x + 1] - cells[cell_index_x])
        
        cells = np.linspace(0, 1, self.feature_map_size[1] + 1)
        cell_index_y = np.argmax(cells > center_y)-1
        offset_y = (center_y - cells[cell_index_y])/(cells[cell_index_y + 1] - cells[cell_index_y])
        
        #Get anchor transformations: h = anchor_h * exp(t_h)
        best_anchor_index = 0 #TODO
        anchor = self.anchors[best_anchor_index]
        
        bbox_height = abs(top - bottom)
        bbox_width = abs(right - left)
        
        t_w = np.log(bbox_width/anchor[0])
        t_h = np.log(bbox_height/anchor[1])
        
        #Put everything into feature feature_map
        feature_map[:5, cell_index_x, cell_index_y] = offset_x, offset_y, t_w, t_h, 1
        feature_map[5 + subj_index, cell_index_x, cell_index_y] = 1
            
        #ToTensor
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        feature_map = transforms.ToTensor()(feature_map)
        
        return {'img': img, 'map': feature_map}