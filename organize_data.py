import random
from collections import defaultdict
from PIL import Image

import os
import json
import argparse

from utils.box_transforms import ResizeBox

def get_folders(path):
    return list(filter(lambda s: os.path.isdir(os.path.join(path, s)), os.listdir(path)))

def train_test_split(train_dir='data/train/', test_dir='data/test/', root_dir='faces/', input_size=320, test_ratio=0.2, random_seed=0):   
    random.seed(random_seed)

    #Create directories for train/cv/test/ data
    for dir in [train_dir, test_dir]:
        if not os.path.exists(dir):
            print('Creating {}'.format(dir))
            os.makedirs(dir)
            
    #Load metadata
    metadata = []

    for dir in get_folders(root_dir):
        with open(root_dir + dir + '/labels/labels.txt', 'r') as f:
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
            img = Image.open(img_path)
            
            rect = metadata[img_path]['rect']
            top, bottom, left, right = rect['top'], rect['bottom'], rect['left'], rect['right']

            #Scale
            size = min(img.height, img.width)
            scale = input_size/size
            img = img.resize((int(img.width*scale), int(img.height*scale)))
            (top, bottom, left, right) = ResizeBox((top, bottom, left, right), (scale,scale))
            
            #Save resized img
            name = img_path.split('/')[-1]
            new_img_path = test_dir + str(i) + '/' + name
            img.save(new_img_path)
            
            test_meta.append({'path':new_img_path, 'subject':i, 'rect':(top, bottom, left, right)})
            
        #Train
        for img_path in images[n_test_images:]:
            
            img = Image.open(img_path)
            
            rect = metadata[img_path]['rect']
            top, bottom, left, right = rect['top'], rect['bottom'], rect['left'], rect['right']

            #Scale
            size = min(img.height, img.width)
            scale = input_size/size
            img = img.resize((int(img.width*scale), int(img.height*scale)))
            (top, bottom, left, right) = ResizeBox((top, bottom, left, right), (scale,scale))
            
            #Save resized img
            name = img_path.split('/')[-1]
            new_img_path = train_dir + str(i) + '/' + name
            img.save(new_img_path)
                    
            train_meta.append({'path':new_img_path, 'subject':i, 'rect':(top, bottom, left, right)})
        
    with open(test_dir + 'labels.txt', 'w+') as fs:
        json.dump(test_meta, fs)
        
    with open(train_dir + 'labels.txt', 'w+') as fs:
        json.dump(train_meta, fs)
        
    with open(root_dir + 'subjects.txt', 'w+') as fs:
        json.dump(subj_meta, fs)

def arg_parse():
    parser = argparse.ArgumentParser(description='Train-test split')
   
    parser.add_argument("--train", dest = 'train_dir', help = "Directory for trainset",
                        default = "data/train/", type = str)    
    parser.add_argument("--test", dest = 'test_dir', help = "Directory for testset",
                        default = "data/test/", type = str)
    parser.add_argument("--root", dest = 'root_dir', help = "Root directory of data",
                        default = "faces/", type = str)    
    parser.add_argument("--reso", dest = 'input_size', help = "Image size",
                        default = 320, type = int)
    parser.add_argument("--ratio", dest = "test_ratio", help = "Split ratio", default = 0.2)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    train_test_split(args.train_dir, args.test_dir, args.root_dir, args.input_size, args.test_ratio)
