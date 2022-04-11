import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, num_classses, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        if torch.cuda.is_available():
          self.alpha = self.alpha.cuda()
        self.gamma = gamma
        self.num_classes = num_classses

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        y = torch.ones(targets.shape)
        if torch.cuda.is_available():
          y = y.cuda()
        targets = torch.where(targets == 0, targets, y)
        
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        
        at = at.view(-1, self.num_classes)
        
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss

class CoteachingLoss(nn.Module):
    def __init__(self, criterion, forget_rate, device) -> None:
        super(CoteachingLoss, self).__init__()
        self.criterion = criterion
        self.forget_rate = forget_rate
        self.device = device
    
    def forward(self, logits, labels):
        logits_1 = logits[0]
        logits_2 = logits[-1]
        loss_1 = self.criterion(logits_1, labels).mean(dim=1)
        ind_1_sorted = torch.argsort(loss_1.data).to(self.device)

        loss_2 = self.criterion(logits_2, labels).mean(dim=1)
        ind_2_sorted = torch.argsort(loss_2.data).to(self.device)

        remember_rate = 1 - self.forget_rate
        num_remember = int(remember_rate * len(ind_1_sorted))

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]

        loss_1_update = self.criterion(logits_1[ind_2_update], labels[ind_2_update]).mean(dim=0).mean(dim=0)
        loss_2_update = self.criterion(logits_2[ind_1_update], labels[ind_1_update]).mean(dim=0).mean(dim=0)

        return [loss_1_update, loss_2_update]