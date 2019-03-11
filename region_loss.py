import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def build_targets(self, pred_boxes, target, num_anchors, num_classes, nH, nW):
        nB = target.size(0)
        nA = num_anchors
        nC = num_classes
        anchor_step = len(self.anchors)/num_anchors
        conf_mask  = torch.ones(nB, nA, nH, nW) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask   = torch.zeros(nB, nA, nH, nW)
        tx         = torch.zeros(nB, nA, nH, nW) 
        ty         = torch.zeros(nB, nA, nH, nW) 
        tw         = torch.zeros(nB, nA, nH, nW) 
        th         = torch.zeros(nB, nA, nH, nW) 
        tconf      = torch.zeros(nB, nA, nH, nW)
        tcls       = torch.zeros(nB, nA, nH, nW) 
        
        nAnchors = nA*nH*nW
        nPixels  = nH*nW
        
        for b in xrange(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            
        #TODO ...
    
    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        
        nB = output.data.size(0)   # Batch size
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        
        # Get all the parameters from output tensor
        # TODO to_device
        output = output.view(nB, nA, (5+nC), nH, nW)       # separate channels, corresponding to different anchors
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        
        # Build boxes from extracted parameters
        # TODO redo
        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4)) #FIXME i don't know why it is transformed like this
        
        coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = self.build_targets(pred_boxes, target.data, nA, nC, nH, nW)
        
        cls_mask = (cls_mask == 1)
        
        tx    = Variable(tx.cuda())    # target x offset
        ty    = Variable(ty.cuda())    # target x offset
        tw    = Variable(tw.cuda())    # target width
        th    = Variable(th.cuda())    # target height
        tconf = Variable(tconf.cuda())                             # target confidence
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())    # target class distribution
        
        coord_mask = Variable(coord_mask.cuda())           # masks coords that contribute into training (have conf==1)
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())    # masks classes to affect only class whose p==1
        cls        = cls[cls_mask].view(-1, nC)
        
        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls