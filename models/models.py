from torchvision.models import resnet34

class ForkNet(torch.nn.Module):
    def __init__(self, num_features=512, num_anchors=1, S=5):
        super().__init__()
        
        self.num_anchors = num_anchors
        
        self.encoder = resnet34(pretrained=True)
		self.pool = nn.AdaptiveAvgPool2d((S, S))
        
        self.reg_head = nn.Conv2d(512, (4+1)*num_anchors, 1)
		
        self.class_head = nn.Sequential(
            nn.Conv2d(512, num_anchors*num_features, 1),
            nn.GroupNorm(num_anchors, num_anchors*num_features), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x) 
        x = self.encoder.layer2(x)        
        x = self.encoder.layer3(x)        
        x = self.encoder.layer4(x)
        
        x = self.pool(x)
        
        x_reg = self.reg_head(x)		#Regression head
        x_class = self.class_head(x)	#Classification head
		
		#Reshape bbox attributes
		nB, _, nH, nW = x_reg.shape
		x_reg = x_reg.view(nB, self.num_anchors, 4+1, nH, nW)      
		x_reg = x_reg.permute(0, 1, 3, 4, 2)
                
        return x_reg, x_class
		
class Classifier(torch.nn.Module):
    def __init__(self, num_classes=1, num_features=512, num_anchors=1):
        super().__init__()
        
        self.num_features = num_features
        self.num_anchors = num_anchors  
        self.metric_fn = nn.Conv3d(num_features, num_classes, 1)
        
    def forward(self, x, target=None):
      
        nB, _, nH, nW = x.shape
        nC = self.num_features
        nA = self.num_anchors
        
        x = x.view(nB, nC, nA, nH, nW)
        x_class = self.metric_fn(x)
        
        x_class = x_class.permute(0, 2, 3, 4, 1)
        return x_class