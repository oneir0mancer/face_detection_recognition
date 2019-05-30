import torch
import torch.nn as nn

import numpy as np
import dlib
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34

class ResNet(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        self.encoder = resnet34(pretrained=True)
        
        self.relu = nn.ReLU(inplace=True) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    
    def forward(self, x):        
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x) 
        x = self.encoder.layer2(x)        
        x = self.encoder.layer3(x)        
        x = self.encoder.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


transform = transforms.Compose([
    transforms.RandomCrop(320),
    transforms.ColorJitter(brightness=.1, hue=.05, saturation=.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = ImageFolder('./data/train/', transform=transform)
testset = ImageFolder('./data/test/', transform=val_transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

resnet = ResNet(num_classes=285).to(device)

lr = 1e-4

from torch.optim.lr_scheduler import StepLR

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
scheduler = StepLR(optimizer, 10)


num_epoch = 20
hist_base = []
info_step = 5
batch_size=128

for epoch in range(num_epoch):
    scheduler.step()
  
    resnet.train()
    running_loss = 0.0
    train_loss = 0.0
    
    for i, (img, target) in enumerate(trainloader):
        optimizer.zero_grad()
        
        output = resnet(img.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loss += loss.item()
        
        if (i+1) % info_step == 0:
            print(' [{} - {}],\ttrain loss: {:.5}'.format(epoch+1, i+1, running_loss/info_step/batch_size))
            running_loss = 0.0
            
    train_loss /= len(trainset)
    print('\n [{}], \ttrain loss: {:.5}'.format(epoch+1, train_loss))

    resnet.eval()
    val_loss = 0.0
    for i, (img, target) in enumerate(testloader):
        with torch.no_grad():
            output = resnet(img.to(device))
            loss  = criterion(output, target.to(device))
        
        val_loss += loss.detach().item()
            
    val_loss /= len(testset)        
    print(' [{}], \tval loss: {:.5}\n'.format(epoch+1, val_loss))
    print()
  
    hist_base.append((train_loss, val_loss))

#Test
from utils.utils import bbox_iou_numpy

testloader = DataLoader(testset, batch_size=1, shuffle=False)
resnet.eval()

detector = dlib.get_frontal_face_detector()

test_transforms = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open('data/test/labels.txt', 'r') as f:
    test_meta = json.load(f) 

no_face = []
wrong = []
mae = 0
iou = 0
acc = 0

#Get accuracy
for i, (img, target) in enumerate(testloader):
    with torch.no_grad():
        output = resnet(img.to(device))
    
    if output.squeeze(0).argmax().item() == target.squeeze(0).item():
        acc +=1
	else
		wrong.append(i)

#Get other stats
start = time.time()
for i, meta in enumerate(test_meta):
    
    img = Image.open(meta['path'])
    img = np.array(img)[:,:,:3]
    img = Image.fromarray(img)
    with torch.no_grad():
        tensor = test_transforms(img).unsqueeze(0)
        output = resnet(tensor.to(device))
    
    faces = list(detector(np.array(img), 1))
    if len(faces) == 0:
        no_face.append(i)
        continue
    
    pred_box = np.array([faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()])
    top, bottom, left, right = meta['rect']
    true_box = np.array([left,top, right,bottom])    
    
    mae += np.sum(np.abs(pred_box-true_box))
    iou += bbox_iou_numpy(pred_box, true_box)
    
start = time.time() - start    

print("No face detected on ", len(no_face), " images:")
print(no_face)
print(len(wrong), " faces misclassified:")
print(wrong)

inter = len(set(no_face).intersection(set(wrong)))	#Number of misclassified images with no face detected
                                                    #(so they are already not counted in accuracy)
print("mae = ", mae/len(testset)*320, " px")
print("iou = ", 100*iou/len(testset), "%")
print("acc = ", 100*(acc - len(no_face) + inter)/len(testset), "%")
print("fps = ", len(testset)/start)  