#TODO sort this out
from utils import bbox_iou

#Optimize: cache grid
def detect(tensor):
    with torch.no_grad():
        x_reg, x_features = net(tensor.to(device))

    x_reg = x_reg.view(nB, nA, (4+1), nH, nW)      
    x_reg = x_reg.permute(0, 1, 3, 4, 2)    #put attributes last

    x_reg[..., 0] = torch.sigmoid(x_reg[..., 0])  # Center x
    x_reg[..., 1] = torch.sigmoid(x_reg[..., 1])  # Center y
    x_reg[..., 4] = torch.sigmoid(x_reg[..., 4])  # Conf

    #TODO calc grid just once
    grid_x = torch.arange(nW, dtype=torch.float32).repeat(nW, 1).view([1, 1, nH, nW]).to(device)
    grid_y = torch.arange(nH, dtype=torch.float32).repeat(nH, 1).t().view([1, 1, nH, nW]).to(device)
    scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors]).to(device)
    anchor_w = scaled_anchors[:, 0].view((1, nA, 1, 1))
    anchor_h = scaled_anchors[:, 1].view((1, nA, 1, 1))

    x_reg[..., 0] = x_reg[..., 0] + grid_x
    x_reg[..., 1] = x_reg[..., 1] + grid_y
    x_reg[..., 2] = torch.exp(x_reg[..., 2]) * anchor_w
    x_reg[..., 3] = torch.exp(x_reg[..., 3]) * anchor_h

    #Add cell index
    index_grid = grid_x + grid_y*nH*nW
    index_tensor = torch.cat((index_grid, nH*nW+index_grid, 2*nH*nW+index_grid), dim=1)
    x_reg = torch.cat((x_reg, index_tensor.unsqueeze(-1)), dim=-1)

    x_reg = x_reg.view(1, nA*nH*nW, 4+1+1)    #x,y,h,w + conf + ind

    # From (x, y, width, height) to (x1, y1, x2, y2)
    box_corner = x_reg.new(x_reg.shape)
    box_corner[:, :, 0] = x_reg[:, :, 0] - x_reg[:, :, 2] / 2
    box_corner[:, :, 1] = x_reg[:, :, 1] - x_reg[:, :, 3] / 2
    box_corner[:, :, 2] = x_reg[:, :, 0] + x_reg[:, :, 2] / 2
    box_corner[:, :, 3] = x_reg[:, :, 1] + x_reg[:, :, 3] / 2
    x_reg[:, :, :4] = box_corner[:, :, :4]

    #Threshold
    prediction = x_reg.squeeze(0)    #let batchsize be 1
    conf_mask = (prediction[:, 4] >= conf_thres).squeeze()
    prediction = prediction[conf_mask]

    if not prediction.size(0):
        return []

    #NMS
    # Sort the detections by maximum objectness confidence
    _, conf_sort_index = torch.sort(prediction[:, 4], descending=True)
    prediction = prediction[conf_sort_index]

    max_detections = []
    while prediction.size(0):
        # Get detection with highest confidence and save as max detection
        max_detections.append(prediction[0].unsqueeze(0))
        # Stop if we're at the last detection
        if len(prediction) == 1:
            break
        # Get the IOUs for all boxes with lower confidence
        ious = bbox_iou(max_detections[-1], prediction[1:])
        # Remove detections with IoU >= NMS threshold
        prediction = prediction[1:][ious < nms_thres]
    
    num_features = 512
    x_features = x_features.view(-1, num_features, nA, nH, nW)
    x_features = x_features.permute(0, 2, 3, 4, 1) 
    
    database = []
    for det in max_detections:
        index = det[..., -1]
        index = int(index[0])    #FIXME
        a = index // (nH*nW)           #anchor
        y = (index % (nH*nW)) // nH    #height on grid
        x = (index % (nH*nW)) % nH     #width on grid

        features = x_features[0, a, x, y, :]
        database.append((det,features))
        
    return database
    
    
#Get database of original photos
orig_img = {
    278:'data/train/278/11.jpeg', #stark
    280: 'data/train/280/2.jpg',  #thor
    281:'data/train/281/2.jpeg',  #loki
    282:'data/train/282/2.jpg',   #spidy
    283:'data/train/283/5.jpg'   #pots
}
orig_feat = {}
for k,v in orig_img.items():
    img2 = Image.open(v)
    img2 = transforms.Resize(320)(img2)
    tensor = test_transforms(img2).unsqueeze(0)
    orig_feat[k] = detect(tensor)[0][1]

#Detect and draw bbox
filepath = '02.jpg'
img = Image.open(filepath)
img = transforms.Resize(320)(img)
tensor = test_transforms(img).unsqueeze(0)

database = detect(tensor)

from PIL import ImageDraw
import torch.nn.functional as F

def dist(a,b):
    #return torch.dist(a,b)
    return 1-F.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0))

draw_img = transforms.CenterCrop(320)(img)
draw = ImageDraw.Draw(draw_img)

for det,feat in database:
    box = det[0,:4].cpu().numpy()/5*320
    draw.rectangle(box)
    
    #FIXME
    subj = -1
    d = 1000
    for k,v in orig_feat.items():
        new_d = dist(feat,v)
        if new_d<d:
            d = new_d
            subj = k
    draw.text(box[:2], str(subj))

del draw
draw_img
