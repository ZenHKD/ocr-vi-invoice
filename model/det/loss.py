import torch
import torch.nn as nn

class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask):
        # pred: (N, 1, H, W)
        # gt: (N, 1, H, W)
        # mask: (N, 1, H, W) (training mask)

        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
 
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
            
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        # pred: (N, 1, H, W)
        # gt: (N, 1, H, W)
        # mask: (N, 1, H, W) (ignore mask)

        pred = pred.squeeze(1)
        gt = gt.squeeze(1)
        mask = mask.squeeze(1)

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss

class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss

class DBLoss(nn.Module):
    def __init__(self, alpha=5.0, beta=10.0, ohem_ratio=3.0):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss()
        self.l1_loss = MaskL1Loss()

    def forward(self, predictions, batch):
        # predictions: {'binary': ..., 'thresh': ..., 'thresh_binary': ...}
        # batch: {'gt': ..., 'mask': ..., 'thresh_map': ..., 'thresh_mask': ...}

        pred_binary = predictions['binary']
        pred_thresh = predictions['thresh']
        pred_thresh_binary = predictions['thresh_binary']

        gt = batch['gt']
        mask = batch['mask']
        thresh_map = batch['thresh_map']
        thresh_mask = batch['thresh_mask']

        l_probability = self.bce_loss(pred_binary, gt, mask)
        l_thresh = self.l1_loss(pred_thresh, thresh_map, thresh_mask)
        l_binary = self.dice_loss(pred_thresh_binary, gt, mask)

        loss = l_probability + self.alpha * l_binary + self.beta * l_thresh
        return loss, {'loss': loss, 'l_prob': l_probability, 'l_binary': l_binary, 'l_thresh': l_thresh}
