from torch.utils.data import DataLoader
from torchvision import transforms

from utils.FaceDataset import FaceDataset
from utils.loss import RegionLoss
from utils.box_transforms import *
from utils.training import train, validate, test
from models.models import *

import json
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Script for training the network.')
   
    parser.add_argument("--train", dest = 'train_labels', help = "Labels for trainset",
                        default = "data/train/labels.txt", type = str)    
    parser.add_argument("--test", dest = 'test_labels', help = "Labels for testset",
                        default = "data/test/labels.txt", type = str)
    parser.add_argument("--net", dest = 'net_weights', help = "Path for saving net weights",
                        default = "./net", type = str)    
    parser.add_argument("--clf", dest = 'clf_weights', help = "Path for saving classifier weights",
                        default = "./clf", type = str)
    parser.add_argument("--classes", dest = 'num_classes', help = "Number of classes",
                        default = 285, type = int)
    parser.add_argument("--anchors", dest = "anchor_sizes", help = "Size of anchors",
                        default = "30,80,120", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution",
                        default = 320, type = int)
    parser.add_argument("--bs", dest = 'batch_size', help = "Batch size",
                        default = 320, type = int)
    parser.add_argument("--epoch", dest = 'num_epoch', help = "Number of epochs",
                        default = 20, type = int)
    return parser.parse_args()

args = arg_parse()

with open(args.train_labels, 'r') as f:
    train_meta = json.load(f)
with open(args.test_labels, 'r') as f:
    test_meta = json.load(f)

#Anchors and classes
anchors=[(int(x), int(x)) for x in args.anchor_sizes.split(',')]
num_anchors = len(anchors)
num_classes = args.num_classes

#Transforms on dataset
box_transform = Compose([
    ResizeWithBox(args.reso),
    RandomCropWithBox(args.reso)
])

val_box_transform = Compose([
    ResizeWithBox(args.reso),
    CenterCropWithBox(args.reso)
])

transform = transforms.Compose([
    transforms.ColorJitter(brightness=.1, hue=.05, saturation=.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Dataset and loader
trainset = FaceDataset(train_meta, box_transform=box_transform, img_transform=transform)
testset = FaceDataset(test_meta, box_transform = val_box_transform, img_transform=val_transform)

batch_size = args.batch_size
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#Network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Initialize network...")
net = ForkNet(num_anchors=num_anchors).to(device)
classifier = Classifier(num_classes=num_classes, num_anchors=num_anchors).to(device)

#Loss & optimizer
lr = 1e-4

criterion = RegionLoss(num_classes=num_classes, anchors=anchors, num_anchors=num_anchors)
optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': classifier.parameters()}], lr=lr)

#LR schedule
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, 10)

print("Start training...")
history = []

net.train()
for epoch in range(args.num_epoch):
    scheduler.step()
    train_loss = train(trainloader, len(trainset), info_step=10)
    val_loss = validate(testloader, len(testset))
    history.append((train_loss, val_loss))
	
print("Test...")
testloader = DataLoader(testset, batch_size=1, shuffle=False)
mae, iou, acc, time_checkpoint = test(testloader)
print("mae = ", mae/set_size*args.reso, " pixels")
print("iou = ", 100*iou/set_size, "%")
print("acc = ", 100*acc/set_size, "%")
print("fps = ", set_size/time_checkpoint)

print("Save weights...")
torch.save(net.state_dict(), args.net_weights)
torch.save(classifier.state_dict(), args.clf_weights)
print("Done")