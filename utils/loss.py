import numpy as np

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import bbox_iou

def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, device):
    #pred_boxes: B x A x H x W x 4
    #pred_conf:  B x A x H x W x 1
    #pred_cls:   B x A x H x W x num_classes
    #target: B x n_obj_on_img x 4+class_index
    
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nH, nW = grid_size

    mask = torch.zeros(nB, nA, nH, nW, dtype=torch.uint8).to(device)
    conf_mask = torch.ones(nB, nA, nH, nW, dtype=torch.uint8).to(device)

    tx = torch.zeros(nB, nA, nH, nW).to(device)  
    ty = torch.zeros(nB, nA, nH, nW).to(device)  
    tw = torch.zeros(nB, nA, nH, nW).to(device)  
    th = torch.zeros(nB, nA, nH, nW).to(device) 
    tconf = torch.zeros(nB, nA, nH, nW).to(device)  
    tcls = torch.zeros(nB, nA, nH, nW, nC).to(device)
    

    nGT = 0         # number of ground truth boxes
    nCorrectPred = 0    # number of correctly predicted boxes
    nCorrectBox = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            nGT += 1
            # Convert to position relative to grid
            gx = (target[b, t, 0] * nW).item()
            gy = (target[b, t, 1] * nH).item()
            gw = (target[b, t, 2] * nW).item()
            gh = (target[b, t, 3] * nH).item()
            # Get grid cell indices
            gi = int(gx)
            gj = int(gy)
            
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0).to(device)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), axis=1)).to(device)
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            
            # Set mask to zero to ignore the boxes with overlap larger than threshold (aka correct boxes)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            
            # Find the best matching anchor box
            best_n = torch.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0).to(device)
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
            
            target_label = target[b, t, 4].item()
            # Calculate iou between ground truth and best matching prediction
            # Check if the box is correct
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            
            if iou > 0.5 and score > 0.5:
                nCorrectBox += 1
                
            if pred_label == target_label and score > 0.5:
                nCorrectPred += 1
                
    return nGT, nCorrectPred, mask, conf_mask, tx, ty, tw, th, tconf, tcls



class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1, ignore_thres=0.5, coord_scale=1, obj_scale=1, noobj_scale=1, class_scale=1):
        super(RegionLoss, self).__init__()
        self.coord_scale = coord_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.class_scale = class_scale
        
        self.ignore_thres = ignore_thres
        
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
      
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.mse_loss = nn.SmoothL1Loss().to(self.device)    #
        self.bce_loss = nn.BCELoss().to(self.device)    #
        self.ce_loss = nn.CrossEntropyLoss().to(self.device) #
    
    def forward(self, x_reg, x_class, target):       
        #stride = self.image_dim / nG   #this rescales anchors for mb multiscale, nG is grid size nH or nW
        stride = 1
        _, _, nH, nW, _ = x_reg.shape
        
        # Get attributes from output tensor
        x = torch.sigmoid(x_reg[..., 0])  # Center x
        y = torch.sigmoid(x_reg[..., 1])  # Center y
        w = x_reg[..., 2]  # Width
        h = x_reg[..., 3]  # Height
        pred_conf = torch.sigmoid(x_reg[..., 4])  # Conf
        pred_cls = x_class   #Class distribution
        
        # Calculate offsets for each grid       
        grid_x = torch.arange(nW, dtype=torch.float32).repeat(nW, 1).view([1, 1, nH, nW]).to(self.device)
        grid_y = torch.arange(nH, dtype=torch.float32).repeat(nH, 1).t().view([1, 1, nH, nW]).to(self.device)
        scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]).to(self.device)
        anchor_w = scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))
        
        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(x_reg[..., :4].shape).to(self.device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        
        # conf = sigma(t_o), tconf = 1
        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes, pred_conf, pred_cls, target, self.anchors,
                                                                                    num_anchors=self.num_anchors, num_classes=self.num_classes, grid_size=(nH,nW), 
                                                                                    ignore_thres=self.ignore_thres, device=self.device)
        
        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask
        
        loss_x = self.coord_scale * self.mse_loss(x[mask], tx[mask])
        loss_y = self.coord_scale * self.mse_loss(y[mask], ty[mask])
        loss_w = self.coord_scale * self.mse_loss(w[mask], tw[mask])
        loss_h = self.coord_scale * self.mse_loss(h[mask], th[mask])
            
        loss_conf = self.noobj_scale * self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + \
                    self.obj_scale * self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
          
        loss_cls = self.class_scale * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
        
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
        return loss, (loss_x.item() + loss_y.item() + loss_w.item() + loss_h.item(), loss_conf.item(), loss_cls.item(), nCorrect)
