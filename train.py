from torch.utils.data import DataLoader
from torchvision import transforms

from utils.FaceDataset import FaceDataset
from utils.loss import RegionLoss
from utils.box_transforms import *
from utils.training import train, validate, test
from models.models import *

import json

train_labels = './data/train/labels.txt'
test_labels = './data/test/labels.txt'
net_weights = ''
clf_weights = ''

with open(train_labels, 'r') as f:
    train_meta = json.load(f)
with open(test_labels, 'r') as f:
    test_meta = json.load(f)

#Anchors and classes
anchors=[(30,30), (80,80), (120,120)]
num_anchors = len(anchors)
num_classes = 285

#Transforms on dataset
box_transform = Compose([
    ResizeWithBox(320),
    RandomCropWithBox(320)
])

val_box_transform = Compose([
    ResizeWithBox(320),
    CenterCropWithBox(320)
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

batch_size = 100
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

#Start training
num_epoch = 20
history = []

print("Start training...")
net.train()
for epoch in range(num_epoch):
    scheduler.step()
    train_loss = train(trainloader, len(trainset), info_step=10)
    val_loss = validate(testloader, len(testset))
    history.append((train_loss, val_loss))
	
print("Test...")
testloader = DataLoader(testset, batch_size=1, shuffle=False)
mae, iou, acc, time_checkpoint = test(testloader)
print("mae = ", mae/set_size*320, " pixels")
print("iou = ", 100*iou/set_size, "%")
print("acc = ", 100*acc/set_size, "%")
print("fps = ", set_size/time_checkpoint)

print("Save weights...")
torch.save(net.state_dict(), net_weights)
torch.save(classifier.state_dict(), clf_weights)
print("Done")