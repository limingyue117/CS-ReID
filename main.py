import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import copy

# import swats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from configs.default import get_config
from data import build_dataloader
from models import build_model, build_classifier
from losses import build_losses, build_losses_clo
from losses import make_loss_with_triplet_entropy_mse, make_loss_ce_triplet
from tools.eval_metrics import evaluate
from tools.utils import AverageMeter, Logger, save_checkpoint, set_seed, save_best_checkpoint
from tools.lr_scheduler import WarmupMultiStepLR
from train import train, train_gcn, train_ske, train_cal, train_ltcc
from test import test, test_nkup, test_prcc, test_ltcc


def parse_option():
    parser = argparse.ArgumentParser(description='Train image-based re-id model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, help="market1501, cuhk03, dukemtmcreid, msmt17")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    if not config.EVAL_MODE:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_test.txt'))
    print("==========\nConfig:{}\n==========".format(config))
    print("Currently using GPU {}".format(config.GPU))
    # Set random seed
    set_seed(config.SEED)

    # Build dataloader
    if config.DATA.DATASET == 'prcc' or config.DATA.DATASET == 'prcc_cal':
        trainloader, queryloader_same, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
        num_classes = dataset.num_train_pids
        num_clo_classes = dataset.num_train_clothes
    elif config.DATA.DATASET == 'nkup':
        trainloader, queryloader, galleryloader_same, galleryloader_diff, galleryloader_ldiff, galleryloader_diff_all, dataset, train_sampler = build_dataloader(config)
        num_classes = dataset.num_train_pids
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
        num_classes = dataset.num_train_pids
        num_clo_classes = dataset.num_train_clothes
    
    # Define a matrix pid2clothes with shape (num_pids, num_clothes). 
    # pid2clothes[i, j] = 1 when j-th clothes belongs to i-th identity. Otherwise, pid2clothes[i, j] = 0.
    pid2clothes = torch.from_numpy(dataset.pid2clothes) 
    
    # Build model
    model, classifier = build_model(config, num_classes)
    model = nn.DataParallel(model).cuda()
    classifier = nn.DataParallel(classifier).cuda()


    if config.DATA.DATASET == 'prcc_cal':
        clothes_classifier = build_classifier(config, num_clo_classes)
        clothes_classifier = nn.DataParallel(clothes_classifier).cuda()

    # Build classification and pairwise loss
    if config.DATA.DATASET == 'prcc_cal':
        criterion_cla, criterion_pair, criterion_clothes, criterion_adv, criterion_ma, criterion_mb, center_criterion = build_losses_clo(config, num_classes)
    else:
        criterion_cla, criterion_pair, criterion_ma, criterion_mb, criterion_mc, center_criterion = build_losses(config, num_classes)

    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())

    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        if config.DATA.DATASET == 'prcc_cal':
            optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'swats':
        optimizer = swats.SWATS(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    
    optimizer_center = optim.SGD(center_criterion.parameters(), lr=0.5)
    
    # Build lr_scheduler
    if config.TRAIN.lr_type == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)
    elif config.TRAIN.lr_type == 'oclr':
        scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=0.0004,total_steps=60, verbose=True)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']


    if config.EVAL_MODE:
        print("Evaluate only")
        test_prcc(config, model, queryloader_same, queryloader, galleryloader, dataset)
        # test(config, model, queryloader, galleryloader, dataset)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        start_train_time = time.time()
        train_sampler.set_model(model, epoch, center_criterion.centers)
        
        if config.DATA.DATASET == 'prcc_cal':
            train_cal(epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
                criterion_clothes, criterion_adv, criterion_ma, criterion_mb, optimizer, optimizer_cc, trainloader, pid2clothes)
        elif config.DATA.DATASET == 'ltcc':
            train_ltcc(config, epoch, model, classifier, criterion_cla, criterion_pair, criterion_ma, criterion_mb, center_criterion, optimizer, optimizer_center, trainloader, pid2clothes)
        else:
            train(config, epoch, model, classifier, criterion_cla, criterion_pair, criterion_ma, criterion_mb, center_criterion, optimizer, optimizer_center, trainloader, pid2clothes)
        train_time += round(time.time() - start_train_time)   
        torch.cuda.empty_cache()     
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            print("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc' or config.DATA.DATASET == 'prcc_cal':
                rank1 = test_prcc(config, model, queryloader_same, queryloader, galleryloader, dataset)
            elif config.DATA.DATASET == 'nkup':
                rank1 = test_nkup(config, model, queryloader, galleryloader_same, galleryloader_diff, galleryloader_ldiff, galleryloader_diff_all, dataset)
            elif config.DATA.DATASET == 'ltcc':
                rank1 = test(config, model, queryloader, galleryloader, dataset)
            else:
                rank1 = test(config, model, queryloader, galleryloader, dataset)
            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
            state_dict = model.state_dict()
            state_dict1 = classifier.state_dict()
            save_best_checkpoint({
                'state_dict': state_dict,
                'class':state_dict1,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        else:
            if config.SAVE_MODE:
                state_dict = model.state_dict()
                state_dict1 = classifier.state_dict()
                save_best_checkpoint({
                    'state_dict': state_dict,
                    'class':state_dict1,
                    'rank1': rank1,
                    'epoch': epoch,
                }, False, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()
        if epoch+1 == 120 or epoch+1 == 90:
            print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))





if __name__ == '__main__':
    config = parse_option()
    main(config)