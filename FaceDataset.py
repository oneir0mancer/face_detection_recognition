import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from box_transforms import *

class FaceDataset(Dataset):
    def __init__(self, metadata, subjects, input_size=(640, 640)):
        
        self.metadata = metadata
        self.subjects = subjects
        self.input_size = input_size
                           
        #print('{} images'.format(len(self.images)))

    def __len__(self):
        return len(self.metadata)
               
    def __getitem__(self, idx):
        #Load image
        img = Image.open(self.metadata[idx]['path'])
        
        np_im = np.array(img)[:,:,:3]    #delete alpha if png
        img = Image.fromarray(np_im)
        
        subj = self.metadata[idx]['subject']
        bbox = self.metadata[idx]['rect']    #(top, bottom, left, right)
        
        #Crop         
        crop_rect = CleverRandomCropArea(bbox, img.size, crop_size=self.input_size)
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
        #subj_index = self.subjects.index(subj['subject'])
        subj_index = subj
        
        gt_boxes[0, :] = center_x, center_y, bbox_height, bbox_width, subj_index
        
        #ToTensor
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        #gt_boxes = torch.tensor(gt_boxes, dtype = torch.float32)
        
        return {'img': img, 'target': gt_boxes}
