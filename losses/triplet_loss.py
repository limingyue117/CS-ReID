import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
torch.set_printoptions(threshold=np.inf)

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'cosine':
            inputs = F.normalize(inputs, p=2, dim=1)
            dist = - torch.mm(inputs, inputs.t())

        # # get positive and negative masks
        # targets = targets.view(-1,1)
        # mask = torch.eq(targets, targets.T).float().cuda()              #相同身份为1，其他为0
        # mask_pos1 = mask - 1                                            #相同身份为0，其他为-1
        
        # # For each anchor, find the hardest positive and negative pairs
        # dist_ap, _ = torch.max((dist + mask_pos1 * 99999999.), dim=1)   #相同身份最远正样本
        # dist_an, _ = torch.min((dist + mask * 99999999.), dim=1)        #不同身份最近负样本

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
    

class MALoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, margin=0.0, distance='euclidean'):
        super().__init__()
        self.distance = distance
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, clothes):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'cosine':
            inputs = F.normalize(inputs, p=2, dim=1)
            dist = 1 - torch.mm(inputs, inputs.t())

        # get positive and negative masks
        targets, clothes = targets.view(-1,1), clothes.view(-1,1)
        mask = torch.eq(targets, targets.T).float().cuda()              #相同身份为1，其他为0
        mask_pos = torch.eq(clothes, clothes.T).float().cuda()          #相同身份相同衣服 为1，其他为0
        mask_pos1 = mask_pos - 1                                        #相同身份相同衣服 为0，其他为-1
        mask_neg = mask - mask_pos                                      #相同身份不同衣服 为1，其他为0
        mask_neg1 = mask_neg - 1                                        #相同身份不同衣服 为0，其他为-1

        # For each anchor, find the hardest positive and negative pairs
        dist_ap, _ = torch.max((dist + mask_neg1 * 99999999.), dim=1)   #相同身份不同衣服最远正样本
        dist_an, _ = torch.max((dist + mask_pos1 * 99999999.), dim=1)   #相同身份相同衣服最远负样本
        

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
    

class MBLoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, margin=0.0, distance='euclidean'):
        super().__init__()
        self.distance = distance
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, clothes):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'cosine':
            inputs = F.normalize(inputs, p=2, dim=1)
            dist = 1 - torch.mm(inputs, inputs.t())

        # get positive and negative masks
        targets, clothes = targets.view(-1,1), clothes.view(-1,1)
        mask = torch.eq(targets, targets.T).float().cuda()              #相同身份任何衣服 为1，其他为0

        mask_pos = torch.eq(clothes, clothes.T).float().cuda()          #相同身份相同衣服 为1，其他为0

        
        mask_pos1 = mask_pos - 1                                        #相同身份相同衣服 为0，其他为-1
        mask_pos2 = 1 - mask_pos                                        #相同身份相同衣服 为0，其他为1

        mask_neg = mask - mask_pos                                      #相同身份不同衣服 为1，其他为0
        mask_neg1 = mask_neg - 1                                        #相同身份不同衣服 为0，其他为-1
        mask_neg2 = 1 - mask_neg                                        #相同身份不同衣服 为0，其他为1
        
        # For each anchor, find the hardest positive and negative pairs
        dist_ap, _ = torch.min((dist + mask_neg2 * 99999999.), dim=1)       #相同身份不同衣服最近正样本
        dist_an, _ = torch.max((dist + mask_pos1 * 99999999.), dim=1)       #相同身份相同衣服最远负样本


        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
    
class MCLoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, margin=0.0, distance='euclidean'):
        super().__init__()
        self.distance = distance
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, clothes):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'cosine':
            inputs = F.normalize(inputs, p=2, dim=1)
            dist = 1 - torch.mm(inputs, inputs.t())

        # get positive and negative masks
        targets, clothes = targets.view(-1,1), clothes.view(-1,1)
        mask = torch.eq(targets, targets.T).float().cuda()              #相同身份任何衣服 为1，其他为0

        mask_pos = torch.eq(clothes, clothes.T).float().cuda()          #相同身份相同衣服 为1，其他为0
        
        mask_pos1 = mask_pos - 1                                        #相同身份相同衣服 为0，其他为-1
        mask_pos2 = 1 - mask_pos                                        #相同身份相同衣服 为0，其他为1

        mask_neg = mask - mask_pos                                      #相同身份不同衣服 为1，其他为0
        mask_neg1 = mask_neg - 1                                        #相同身份不同衣服 为0，其他为-1
        mask_neg2 = 1 - mask_neg                                        #相同身份不同衣服 为0，其他为1
        
        # For each anchor, find the hardest positive and negative pairs
        dist_ap, _ = torch.min((dist + mask_neg2 * 99999999.), dim=1)       #相同身份不同衣服最近正样本
        dist_an, _ = torch.min((dist + mask_pos2 * 99999999.), dim=1)       #相同身份相同衣服最近负样本


        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss