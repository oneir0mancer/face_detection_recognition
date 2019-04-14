import random
from collections import defaultdict
from PIL import Image

import os
import json

from box_transforms import *

def train_test_split(train_dir='data/train/', test_dir='data/test/', faces_dir='./', input_size=320, test_ratio=0.2, random_seed=0):   
    random.seed(random_seed)

    #Create directories for train/cv/test/ data
    for dir in [train_dir, test_dir]:
        if not os.path.exists(dir):
            print('Creating {}'.format(dir))
            os.makedirs(dir)
            
    #Load metadata
    metadata = []

    for dir in ['fei', 'caltech_faces', 'gt_db', 'mine']:
        with open(faces_dir + 'faces/' + dir + '/labels/labels.txt', 'r') as f:
            metadata.extend(json.load(f))

    #Transform to dict for convenience
    metadata = {img['path']: img['subjects'][0] for img in metadata}
            
    #Get images for all subjects
    subjects = defaultdict(list)
    for image_path in metadata.keys():
        subj = metadata[image_path]['subject']
        subjects[subj].append(image_path)
        

    #Train-test split
    subj_meta = {}
    test_meta = []
    train_meta = []

    for i, (subj, images) in enumerate(subjects.items()):
        subj_meta[i] = subj
        
        for dir in [train_dir, test_dir]:
            path = dir + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
        
        n_images = len(images)
        n_test_images = max(1, int(n_images * test_ratio))
        random.shuffle(images)
        
        #Test
        for img_path in images[:n_test_images]:
            img = Image.open(faces_dir + img_path)
            
            rect = metadata[img_path]['rect']
            top, bottom, left, right = rect['top'], rect['bottom'], rect['left'], rect['right']

            #Scale
            size = min(img.height, img.width)
            scale = input_size/size
            img = img.resize((int(img.width*scale), int(img.height*scale)))
            (top, bottom, left, right) = Resize((top, bottom, left, right), (scale,scale))
            
            #Save resized img
            name = img_path.split('/')[-1]
            new_img_path = test_dir + str(i) + '/' + name
            img.save(new_img_path)
            
            test_meta.append({'path':new_img_path, 'subject':i, 'rect':(top, bottom, left, right)})
            
        #Train
        for img_path in images[n_test_images:]:
            
            img = Image.open(faces_dir + img_path)
            
            rect = metadata[img_path]['rect']
            top, bottom, left, right = rect['top'], rect['bottom'], rect['left'], rect['right']

            #Scale
            size = min(img.height, img.width)
            scale = input_size/size
            img = img.resize((int(img.width*scale), int(img.height*scale)))
            (top, bottom, left, right) = Resize((top, bottom, left, right), (scale,scale))
            
            #Save resized img
            name = img_path.split('/')[-1]
            new_img_path = train_dir + str(i) + '/' + name
            img.save(new_img_path)
                    
            train_meta.append({'path':new_img_path, 'subject':i, 'rect':(top, bottom, left, right)})
        
    with open(test_dir + 'labels.txt', 'w+') as fs:
        json.dump(test_meta, fs)
        
    with open(train_dir + 'labels.txt', 'w+') as fs:
        json.dump(train_meta, fs)
        
    with open(faces_dir + 'subjects.txt', 'w+') as fs:
        json.dump(subj_meta, fs)

if __name__ == "__main__":
    train_test_split()