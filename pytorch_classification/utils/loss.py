import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss:
    def __init__(self, loss=None, weights=None, alpha=0.25, gamma=2):
        self.loss = loss
        self.weights = weights
        self.alpha = alpha
        self.gamma = gamma
    
    def get_loss(self):
        if self.loss=='BCE':
            criterion = nn.BCEWithLogitsLoss()
        elif self.loss=='WBCE':
            wt = self.weights[1]/self.weights[0]
            criterion = nn.BCEWithLogitsLoss(pos_weight=wt)
        elif self.loss=='FC':
            criterion = FocalLoss(alpha=self.alpha,gamma=self.gamma)
        elif self.loss=='CE':
            criterion = nn.CrossEntropyLoss()
        elif self.loss=='WCE':
            criterion = nn.CrossEntropyLoss(self.weights)

        return criterion

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss