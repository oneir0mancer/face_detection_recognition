import torch
import math
import torch.nn as nn
import torch.nn.functional as F

#TODO move to utils
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres):
    #pred_boxes: B x A x H x W x 4
    #pred_conf:  B x A x H x W x 1
    #pred_cls:   B x A x H x W x num_classes
    #target: B x n_obj_on_img x 4+class_index
    
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nH, nW = grid_size

    mask = torch.zeros(nB, nA, nH, nW, requers_graph=True)
    conf_mask = torch.ones(nB, nA, nH, nW, requers_graph=True)

    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)

    nGT = 0         # number of ground truth boxes
    nCorrect = 0    # number of correctly predicted boxes
    for b in range(nB):
        for t in range(target.shape[1]):
            nGT += 1
            # Convert to position relative to grid
            gx = target[b, t, 0] * nW
            gy = target[b, t, 1] * nH
            gw = target[b, t, 2] * nW
            gh = target[b, t, 3] * nH
            # Get grid cell indices
            gi = int(gx)
            gj = int(gy)
            
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), axis=1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            
            # Set mask to zero to ignore the boxes with overlap larger than threshold (aka correct boxes)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            
            # One-hot encoding of label
            target_class = int(target[b, t, 4])
            tcls[b, best_n, gj, gi, target_class] = 1
            tconf[b, best_n, gj, gi] = 1
            
            
            # Calculate iou between ground truth and best matching prediction
            # Check if the box is correct
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1
                
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.coord_scale = 1
        self.obj_scale = 1
        self.noobj_scale = 1
        self.class_scale = 1
        
        self.ignore_thres = 0.5
      
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.mse_loss = nn.MSELoss().to(self.device)
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
    
    def forward(self, output, target):
        #output : B x A*(4+1+num_classes) x H x W
        
        device = self.device
        #stride = self.image_dim / nG   #this rescales anchors for mb multiscale, nG is grid size nH or nW
        stride = 1
        
        nB, _ , nH, nW = output.shape
        nA = self.num_anchors
        nC = self.num_classes
        
        output = output.view(nB, nA, (4+1+nC), nH, nW)         # reshape for convenience
        output = output.permute(0, 1, 3, 4, 2).contiguous()    # Get bbox_attr dimention to be the last
        
        # Get attributes from output tensor
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls distribution
        
        # Calculate offsets for each grid       
        grid_x = torch.arange(nW).repeat(nW, 1).view([1, 1, nH, nW]).to(device)
        grid_y = torch.arange(nH).repeat(nH, 1).t().view([1, 1, nH, nW]).to(device)
        scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]).to(device)
        anchor_w = scaled_anchors[:, 0].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1].view((1, nA, 1, 1))
        
        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape).to(device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        
        # conf = sigma(t_o), tconf = 1
        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes, pred_conf, pred_cls, target, anchors,
                                                                                    num_anchors=nA, num_classes=nC, grid_size=(nH,nW), ignore_thres=self.ignore_thres)
        
        nProposals = int((pred_conf > 0.5).sum().item())
        recall = float(nCorrect / nGT) if nGT else 1
        precision = float(nCorrect / nProposals)
        
        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask
        
        loss_x = self.coord_scale * self.mse_loss(x[mask], tx[mask])
        loss_y = self.coord_scale * self.mse_loss(y[mask], ty[mask])
        loss_w = self.coord_scale * self.mse_loss(w[mask], tw[mask])    #TODO sqrt
        loss_h = self.coord_scale * self.mse_loss(h[mask], th[mask])    #TODO sqrt
        
        loss_conf = self.noobj_scale * self.mse_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + \
                    self.obj_scale * self.mse_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
          
        loss_cls = self.class_scale * self.mse_loss(cls, tcls)
        
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
        return loss, (loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall, precision)
