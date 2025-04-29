import torch
from torch import nn

from losses.cross_entropy_label_smooth import CrossEntropyLabelSmooth
from losses.triplet_loss import TripletLoss, MALoss, MBLoss, MCLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.center_loss import CenterLoss
from losses.clothes_based_adversarial_loss import ClothesBasedAdversarialLoss

def build_losses(config, num_classes):

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=config.MODEL.FEATURE_DIM, use_gpu=True)  # center loss
    # Build classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M, distance=config.TEST.DISTANCE)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))
    
    criterion_ma = MALoss(margin=config.LOSS.MA, distance=config.TEST.DISTANCE)
    criterion_mb = MBLoss(distance=config.TEST.DISTANCE)
    criterion_mc = MCLoss(distance=config.TEST.DISTANCE)

    return criterion_cla, criterion_pair, criterion_ma, criterion_mb, criterion_mc, center_criterion

def build_losses_clo(config, num_classes):

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=config.MODEL.FEATURE_DIM, use_gpu=True)  # center loss
    # Build classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M, distance=config.TEST.DISTANCE)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))
    
    criterion_clothes = CosFaceLoss(scale=config.LOSS.CLA_S, margin=0)
    criterion_cal = ClothesBasedAdversarialLoss(scale=config.LOSS.CLA_S, epsilon=0.1)

    criterion_ma = MALoss(distance=config.TEST.DISTANCE)
    criterion_mb = MBLoss(distance=config.TEST.DISTANCE)

    return criterion_cla, criterion_pair, criterion_clothes, criterion_cal, criterion_ma, criterion_mb, center_criterion

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    # loss /= len(xs)
    return loss

def make_loss_ce_triplet(cfg):    # modified by gu
    triplet = TripletLoss(cfg.LOSS.PAIR_M)  # triplet loss
    xent = CrossEntropyLabelSmooth()

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)  # 3.2403
        loss = loss_x + loss_t  # 11.1710
        return loss
    return loss_func

def make_loss_with_triplet_entropy_mse(cfg):    # modified by gu
    criterion_pair = TripletLoss(margin=cfg.LOSS.PAIR_M)  # triplet loss
    criterion_cla = CrossEntropyLabelSmooth()
    criterion_mse = FeatLoss()

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        B = int(score.shape[0 ] / 2)
        loss_x = criterion_cla(score, target)         # 6.618
        loss_t = criterion_pair(feat, target)   # 3.2403
        loss_f = criterion_mse(feat[0: B], feat[B:])      # 0.99
        loss = loss_x + loss_t + loss_f        # 11.1710
        return loss

    return criterion_cla, criterion_pair, criterion_mse, loss_func

class FeatLoss(nn.Module):
    def __init__(self, ):
        super(FeatLoss, self).__init__()

    def forward(self, feat1, feat2):    # [64, 2048], [64, 2048]
        B, C = feat1.shape

        dist = torch.pow(torch.abs(feat1 - feat2), 2).sum(dim=-1)

        loss = (1. / (1. + torch.exp(-dist))).mean()

        # loss = dist.mean()

        return loss