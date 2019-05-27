import os
import os.path
import json
import argparse

import dlib
from skimage import io

IMAGE_FORMATS = {'.jpg', '.jpeg', '.png'}

def get_images(path):
    return list(filter(lambda s: os.path.isfile(os.path.join(path, s)) and 
                       os.path.splitext(s)[1] in IMAGE_FORMATS, os.listdir(path)))
def get_folders(path):
    return list(filter(lambda s: os.path.isdir(os.path.join(path, s)), os.listdir(path)))

#Entry = ['subject', 'name', 'path'])
def grab_db_plain(path, divisor):
    res = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        ext = os.path.splitext(file)[1]
        if os.path.isfile(file_path) and ext in IMAGE_FORMATS:
            subject, name = file.split(divisor)
            res.append((path + subject, name, file_path))
    return res


def grab_db_folders(path):
    res = []
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                ext = os.path.splitext(file)[1]
                if os.path.isfile(file_path) and ext in IMAGE_FORMATS:
                    res.append((path + dir, file, file_path))
    return res

def get_entry_subjects(entries):
    subjects = []
    for k in entries.keys():
        subjects.extend(list(set(map(lambda e: e[0], entries[k]))))
    return subjects

def label_bboxes_on_images(dirs=['fei', 'caltech_faces', 'gt_db']):
    detector = dlib.get_frontal_face_detector()
    no_faces = []

    for dir in dirs:
        label_path = base_dir + dir + '/labels/'
        metadata = []

        for (subject, photo, filename) in entries[dir]:
            img = io.imread(filename)
            img = img[:,:,:3]    #remove alpha channel

            faces = list(detector(img, 1))

            if len(faces) == 0:
                no_faces.append(filename)
                continue

            face = faces[0]
            rect_meta = {'top':face.top(), 'bottom':face.bottom(), 
                         'right':face.right(), 'left':face.left()}
            full_meta = {'path':filename, 
                         'subjects':[ {'subject':subject, 'rect':rect_meta} ] }

            metadata.append(full_meta)

        with open(label_path + 'labels.txt', 'w+') as fs:
            json.dump(metadata, fs)

    print("Not detected on {} images".format(len(no_faces)))
    return no_faces

def arg_parse():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument("--root", dest = 'base_dir', help = "Root directory",
                        default = "faces/", type = str)    
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    base_dir = args.base_dir

    print("Create directories for label data")
    for dir in get_folders(base_dir):
        path = base_dir + dir + 'labels/'
        if not os.path.exists(path):
            print('Creating {}'.format(path))
            os.makedirs(path)

    print("Get all images from data folders...")
    entries = {
        'fei': grab_db_plain(base_dir + 'fei/', '-'),
        'caltech_faces': grab_db_folders(base_dir + 'caltech_faces/'),
        'gt_db': grab_db_folders(base_dir + 'gt_db/')
        #'mine': grab_db_folders(base_dir + 'mine/')
    }

    print("Label images...")
    no_faces = label_bboxes_on_images()
    print(no_faces)