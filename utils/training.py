def train(trainloader, set_size, info_step=100):
    net.train()
    running_loss = 0.0
    train_loss = 0.0
    
    losses = [0 for i in range(4)]
    running_losses = [0 for i in range(4)]
    
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()
        
        x_reg, x_features = net(batch['img'].to(device))
        x_class = classifier(x_features, batch['target'].to(device))
        
        loss, info = criterion(x_reg, x_class, batch['target'].to(device))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loss += loss.item()
        
        for j in range(len(losses)):
            losses[j] += info[j]
            running_losses[j] += info[j]
        
        if (i+1) % info_step == 0:
            print(' [{} - {}],\ttrain loss: {:.5}'.format(epoch+1, i+1, running_loss/info_step/batch_size))
            running_loss = 0.0
            
            for j in range(len(losses)):
                running_losses[j] /= info_step*batch_size
            print(' coord loss: {:.5} \tobj loss: {:.5} \tclass loss: {:.5} \tacc: {:.5}'.format(*running_losses))
            running_losses = [0 for j in range(4)]
            
    train_loss /= set_size
    
    for i in range(len(losses)):
        losses[i] /= set_size
        
    print('\n [{}], \ttrain loss: {:.5}'.format(epoch+1, train_loss))
    print(' coord loss: {:.5} \tobj loss: {:.5} \tclass loss: {:.5} \tacc: {:.5}'.format(*losses))
    return train_loss
  

def validate(valloader, set_size):
    net.eval()
    losses = [0 for i in range(4)]
    val_loss = 0.0
    for i, batch in enumerate(valloader):
        with torch.no_grad():
            x_reg, x_features = net(batch['img'].to(device))
            x_class = classifier(x_features, batch['target'].to(device))
            loss, info  = criterion(x_reg, x_class, batch['target'].to(device))
        
        val_loss += loss.detach().item()
        for j in range(len(losses)):
            losses[j] += info[j]
            
    val_loss /= set_size
    for i in range(len(losses)):
        losses[i] /= set_size
        
    print(' [{}], \tval loss: {:.5}'.format(epoch+1, val_loss))
    print(' coord loss: {:.5} \tobj loss: {:.5} \tclass loss: {:.5} \tacc: {:.5}'.format(*losses))
    print()
    return val_loss
	

import time
from utils import non_max_suppression, bbox_iou_numpy

def test(testloader):
	net.eval()

	no_face = []
	wrong = []
	mae = 0
	iou = 0
	acc = 0

	time_checkpoint = time.time()
	for i,batch in enumerate(testloader):
		with torch.no_grad():
			x_reg, x_features = net(batch['img'].to(device))
			x_class = classifier(x_features, batch['target'].to(device))
			
			nB,_,nH,nW = x_reg.size()
			nC = 285
			nA = 3

			anchors = anchors
			stride = 1
			
			prediction = torch.cat((x_reg, x_class), dim=-1)

			# Get attributes from output tensor
			prediction[..., 0] = torch.sigmoid(prediction[..., 0])  # Center x
			prediction[..., 1] = torch.sigmoid(prediction[..., 1])  # Center y
			prediction[..., 4] = torch.sigmoid(prediction[..., 4])  # Conf
			prediction[..., 5:] = torch.softmax(prediction[..., 5:], dim=-1)  # Cls distribution

			# Calculate offsets for each grid       
			grid_x = torch.arange(nW, dtype=torch.float32).repeat(nW, 1).view([1, 1, nH, nW]).to(device)
			grid_y = torch.arange(nH, dtype=torch.float32).repeat(nH, 1).t().view([1, 1, nH, nW]).to(device)
			scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors]).to(device)
			anchor_w = scaled_anchors[:, 0].view((1, nA, 1, 1))
			anchor_h = scaled_anchors[:, 1].view((1, nA, 1, 1))

			# Add offset and scale with anchors
			prediction[..., 0] = prediction[..., 0] + grid_x
			prediction[..., 1] = prediction[..., 1] + grid_y
			prediction[..., 2] = torch.exp(prediction[..., 2]) * anchor_w
			prediction[..., 3] = torch.exp(prediction[..., 3]) * anchor_h

			prediction = prediction.view(nB, nA*nH*nW, 4+1+nC)
			pred = non_max_suppression(prediction, nC)[0]
			if pred is None:
				no_face.append(i)
				continue
			
			#FIXME
			#TODO clamp
			pred_boxes = pred[:, :4].cpu().numpy()
			scores = pred[:, 4].cpu().numpy()
			pred_labels = pred[:, -1].cpu().numpy()

			sort_i = np.argsort(scores)
			pred_labels = pred_labels[sort_i]
			pred_boxes = pred_boxes[sort_i]
			
			pred_box = pred_boxes[-1]/5
			
			true_box = batch['target'][0][-1][:4].cpu().numpy()
			tb_x1, tb_x2 = true_box[0] - true_box[2] / 2, true_box[0] + true_box[2] / 2
			tb_y1, tb_y2 = true_box[1] - true_box[3] / 2, true_box[1] + true_box[3] / 2
			true_box = np.array([tb_x1, tb_y1, tb_x2, tb_y2])
			
			mae += np.sum(np.abs(pred_box-true_box))
			iou += bbox_iou_numpy(pred_box, true_box)
			
			
			label = pred_labels[-1]
			true_label = batch['target'][0][0][-1].item()
			
			if label==true_label:
				acc+=1
			else:
				wrong.append(i)
				
	time_checkpoint = time.time() - time_checkpoint

	print("Not detected on ", len(no_face), " images")
	print(no_face)
	print(len(wrong), " images misclassified")
	print(wrong)
	return mae, iou, acc, time_checkpoint