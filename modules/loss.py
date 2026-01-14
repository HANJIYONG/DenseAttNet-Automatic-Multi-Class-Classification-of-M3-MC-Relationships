from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False,reduction='none', reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.ce = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, inputs, targets):#nn.CrossEntropyLoss()
    
        ce_loss = self.ce(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
def dice_loss(input_, target_):
    smooth = 1.
    loss = 0.
    n_classes = input_.shape[1]
    input_ = input_.softmax(dim=1)
    for c in range(n_classes):
        iflat = input_[:, c ].reshape(-1)
        tflat = target_[:, c].reshape(-1)
        intersection = (iflat * tflat).sum()
        
        # w = class_weights[c]
        w=1
        loss += w*(1 - ((2. * intersection + smooth) /
                        (iflat.sum() + tflat.sum() + smooth)))
    return loss


class DiceChannelLoss(nn.Module):
    def __init__(self):
        super(DiceChannelLoss, self).__init__()

    def forward(self, pred, target, smooth=1e-9 ,weights_apply=False):
        
        pred = F.softmax(pred,dim=1) # batch,channel,h,w
        
        num_channels = pred.shape[1]
        dice = torch.zeros(num_channels, device=pred.device)
        
        for i in range(num_channels):
            pred_channel = pred[:, i, :, :]
            target_channel = target[:, i, :, :]
            intersection = (pred_channel * target_channel).sum(dim=(0, 1, 2))
            dice_coeff = (2. * intersection + smooth) / (pred_channel.sum(dim=(0, 1, 2)) + target_channel.sum(dim=(0, 1, 2)) + smooth)
            dice[i] = 1 - dice_coeff

        # Apply weight to the Dice Loss based on epoch_dice value
        if weights_apply:
            weights = (dice/torch.sum(dice))
            dice = dice * weights.to(pred.device)

        dice_loss = torch.exp(dice).sum() # weight automatically
        
        del pred,pred_channel,target_channel,intersection,dice_coeff
        torch.cuda.empty_cache()
        
        return dice, dice_loss
    

class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        super(SoftSkeletonRecallLoss, self).__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]
    
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        rec = rec.mean()
        return -rec
