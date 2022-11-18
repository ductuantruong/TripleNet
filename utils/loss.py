import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conf_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def _hard_negative_mining(self, loss, pos, neg, k):
        loss = loss.detach()
        rank = (loss * (-1 * neg.float())).sort(dim=1)[1].sort(dim=1)[1]
        hard_neg = rank < (pos.long().sum(dim=1, keepdim=True) * k)
        return hard_neg

    def forward(self, xloc, xconf, loc, label, k=3):   # xconf is logits
        pos = label > 0
        neg = label == 0
        label = label.clamp(min=0)
    
        pos_idx = pos.unsqueeze(-1).expand_as(xloc)
        loc_loss = F.smooth_l1_loss(xloc[pos_idx].view(-1, 4), loc[pos_idx].view(-1, 4), 
                                    size_average=False) 


        conf_loss = self.conf_loss(xconf.transpose(1, 2), label)
        hard_neg = self._hard_negative_mining(conf_loss, pos, neg, k)
        conf_loss = conf_loss * (pos + hard_neg).gt(0).float()
        conf_loss = conf_loss.sum()

        N = pos.data.float().sum() + 1e-3
        return loc_loss / N, conf_loss / N
