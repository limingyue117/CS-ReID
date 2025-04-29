import time
import datetime
import logging
import torch
from torch.cuda import amp
from tools.utils import AverageMeter
import numpy as np
import copy



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr
    
def train(config, epoch, model, classifier, criterion_cla, criterion_pair, criterion_ma, criterion_mb, center_criterion, optimizer, optimizer_center, trainloader, pid2clothes):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_ma_loss = AverageMeter()
    batch_mb_loss = AverageMeter()
    batch_center_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()


    lr = float(get_lr(optimizer))
    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, _, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        ma_loss = criterion_ma(features, pids, clothes_ids)
        mb_loss = criterion_mb(features, pids, clothes_ids)


        # if epoch <= 10:
        #     loss = cla_loss + pair_loss + (ma_loss + mb_loss) * epoch / 10.0
        # else:
        #     loss = cla_loss + pair_loss + ma_loss + mb_loss
        loss = cla_loss + pair_loss + ma_loss + mb_loss
        
        if config.LOSS.CENTER_LOSS:
            center_loss = center_criterion(features.detach(), pids) 
            loss_all = loss + center_loss * config.LOSS.CENTER_LOSS_WEIGHT
            loss_all.backward()
            for param in center_criterion.parameters():
                param.grad.data *= (1./ config.LOSS.CENTER_LOSS_WEIGHT)
            optimizer_center.step()
            optimizer.step()
            batch_center_loss.update(center_loss.item(), pids.size(0))
        else:
            # Backward + Optimizer
            loss.backward()
            optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_ma_loss.update(ma_loss.item(), pids.size(0))
        batch_mb_loss.update(mb_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
            'Time:{batch_time.sum:.1f}s '
            'Data:{data_time.sum:.1f}s '
            'ClaLoss:{cla_loss.avg:.4f} '
            'PairLoss:{pair_loss.avg:.4f} ' 
            'MaLoss:{ma_loss.avg:.4f} ' 
            'MbLoss:{mb_loss.avg:.4f} ' 
            'CenterLoss:{center_loss.avg:.4f} ' 
            'Acc:{acc.avg:.2%} '
            'Lr: {lr:.2e}'.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, ma_loss=batch_ma_loss, 
            mb_loss=batch_mb_loss, center_loss=batch_center_loss, acc=corrects, lr=lr))
    

def train_ltcc(config, epoch, model, classifier, criterion_cla, criterion_pair, criterion_ma, criterion_mb, center_criterion, optimizer, optimizer_center, trainloader, pid2clothes):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_ma_loss = AverageMeter()
    batch_mb_loss = AverageMeter()
    batch_center_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()


    lr = float(get_lr(optimizer))
    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, _, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # print('dd')
        # print(pids)
        # print(clothes_ids)
        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        ma_loss = criterion_ma(features, pids, clothes_ids)
        mb_loss = criterion_mb(features, pids, clothes_ids)


        # if epoch <= 30:
        #     loss = cla_loss + pair_loss + (ma_loss + mb_loss) * epoch / 30.0
        # else:
        #     loss = cla_loss + pair_loss + ma_loss + mb_loss
        # if epoch <= 20:
        #     loss = cla_loss + pair_loss
        # else:
        #     loss = cla_loss + pair_loss + ma_loss + mb_loss
        loss = cla_loss + pair_loss
        
        if config.LOSS.CENTER_LOSS:
            center_loss = center_criterion(features.detach(), pids) 
            loss_all = loss + center_loss * config.LOSS.CENTER_LOSS_WEIGHT
            loss_all.backward()
            for param in center_criterion.parameters():
                param.grad.data *= (2./ config.LOSS.CENTER_LOSS_WEIGHT)
            optimizer_center.step()
            optimizer.step()
            batch_center_loss.update(center_loss.item(), pids.size(0))
        else:
            # Backward + Optimizer
            loss.backward()
            optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_ma_loss.update(ma_loss.item(), pids.size(0))
        batch_mb_loss.update(mb_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} ' 
          'MaLoss:{ma_loss.avg:.4f} ' 
          'MbLoss:{mb_loss.avg:.4f} ' 
          'CenterLoss:{center_loss.avg:.4f} ' 
          'Acc:{acc.avg:.2%} '
          'Lr: {lr:.2e}'.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, ma_loss=batch_ma_loss, 
           mb_loss=batch_mb_loss, center_loss=batch_center_loss, acc=corrects, lr=lr))
    
def train_cal(epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, criterion_ma, criterion_mb, optimizer, optimizer_cc, trainloader, pid2clothes):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    batch_ma_loss = AverageMeter()
    batch_mb_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()


    lr = float(get_lr(optimizer))
    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Update the clothes discriminator
        pred_clothes = clothes_classifier(features.detach())   #反向传播中不计算
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        if epoch >= 25:
            optimizer_cc.zero_grad()
            clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        ma_loss = criterion_ma(features, pids, clothes_ids)
        mb_loss = criterion_mb(features, pids, clothes_ids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)

        if epoch <= 40:
            loss = cla_loss + pair_loss
        elif epoch > 40:
            loss = cla_loss + pair_loss + ma_loss + adv_loss
        else:
            loss = cla_loss + pair_loss + ma_loss
        
        # if epoch >= 25:
        #     loss = cla_loss + adv_loss 
        # else:
        #     loss = cla_loss

        loss.backward()
        optimizer.step()


        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        batch_ma_loss.update(ma_loss.item(), pids.size(0))
        batch_mb_loss.update(mb_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} ' 
          'CloLoss:{clo_loss.avg:.4f} '
          'AdvLoss:{adv_loss.avg:.4f} '
          'MaLoss:{ma_loss.avg:.4f} ' 
          'MbLoss:{mb_loss.avg:.4f} ' 
          'Acc:{acc.avg:.2%} '
          'CloAcc:{clo_acc.avg:.2%} '
          'Lr: {lr:.2e}'.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
           clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
           ma_loss=batch_ma_loss, mb_loss=batch_mb_loss,
           acc=corrects, clo_acc=clothes_corrects, lr=lr))


def train_ske(epoch, model, model1, classifier, classifier1, classifier2, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_cla_loss1 = AverageMeter()
    batch_pair_loss1 = AverageMeter()
    batch_cla_loss2 = AverageMeter()
    batch_pair_loss2 = AverageMeter()
    corrects = AverageMeter()
    corrects1 = AverageMeter()
    corrects2 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    model1.train()
    classifier.train()
    classifier1.train()
    classifier2.train()

    end = time.time()
    for batch_idx, (imgs, skes, pids, _) in enumerate(trainloader):
        imgs, skes, pids = imgs.cuda(), skes.cuda(), pids.cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        # Forward
        features1 = model1(skes)
        outputs1 = classifier1(features1)
        _, preds1 = torch.max(outputs1.data, 1)
        # Compute loss
        cla_loss1 = criterion_cla(outputs1, pids)
        pair_loss1 = criterion_pair(features1, pids)
        
        # Forward
        if epoch > 25:
            features2 = torch.cat([features,features1],1)
            outputs2 = classifier2(features2)
            _, preds2 = torch.max(outputs2.data, 1)
            # Compute loss
            cla_loss2 = criterion_cla(outputs2, pids)
            pair_loss2 = criterion_pair(features2, pids)
            loss = cla_loss + pair_loss + cla_loss1 + pair_loss1 + cla_loss2 + pair_loss2
        else:
            features2 = torch.cat([features,features1],1)
            outputs2 = classifier2(features2.detach())
            _, preds2 = torch.max(outputs2.data, 1)
            # Compute loss
            cla_loss2 = criterion_cla(outputs2, pids)
            pair_loss2 = 0
            loss = cla_loss + pair_loss + cla_loss1 + pair_loss1 + cla_loss2
        
        
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        corrects1.update(torch.sum(preds1 == pids.data).float()/pids.size(0), pids.size(0))
        corrects2.update(torch.sum(preds2 == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_cla_loss1.update(cla_loss1.item(), pids.size(0))
        batch_pair_loss1.update(pair_loss1.item(), pids.size(0))
        batch_cla_loss2.update(cla_loss2.item(), pids.size(0))
        batch_pair_loss2.update(pair_loss2.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '
          'skeClaLoss:{cla_loss1.avg:.4f} '
          'skePairLoss:{pair_loss1.avg:.4f} '
          'skeAcc:{acc1.avg:.2%} '
          'allClaLoss:{cla_loss2.avg:.4f} '
          'allPairLoss:{pair_loss2.avg:.4f} '
          'allAcc:{acc2.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects,
           cla_loss1=batch_cla_loss1, pair_loss1=batch_pair_loss1, acc1=corrects1,
           cla_loss2=batch_cla_loss2, pair_loss2=batch_pair_loss2, acc2=corrects2))
    
def train_gcn(epoch, model, classifier, criterion_cla, criterion_pair, criterion_mse, loss_func, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_mse_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, _, mask) in enumerate(trainloader):
        imgs, pids = imgs.cuda(), pids.cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()

        b, c, h, w = imgs.shape

        mask = mask.cuda()
        mask_i = mask.argmax(dim=1).unsqueeze(dim=1)        # [64, 1, 256, 128]
        mask_i = mask_i.expand_as(imgs)
        img_a = copy.deepcopy(imgs)


        # upper clothes sampling
        index = np.random.permutation(b)
        img_r = imgs[index]                                  # [64, 3, 256, 128]
        msk_r = mask_i[index]                               # [64, 6, 256, 128]
        img_a[mask_i == 2] = img_r[msk_r == 2]

        # pant sampling
        index = np.random.permutation(b)
        img_r = imgs[index]  # [64, 3, 256, 128]
        msk_r = mask_i[index]  # [64, 6, 256, 128]
        img_a[mask_i == 3] = img_r[msk_r == 3]

        img_c = torch.cat([imgs, img_a], dim=0)
        target_c = torch.cat([pids, pids], dim=0)

        # Forward
        features = model(img_c)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, target_c)
        pair_loss = criterion_pair(features, target_c)
        B = int(outputs.shape[0] / 2)
        mse_loss = criterion_mse(features[0: B], features[B:])
        loss = cla_loss + pair_loss   
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == target_c.data).float()/target_c.size(0), target_c.size(0))
        batch_cla_loss.update(cla_loss.item(), target_c.size(0))
        batch_pair_loss.update(pair_loss.item(), target_c.size(0))
        batch_mse_loss.update(mse_loss.item(), target_c.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Cla_Loss:{claloss.avg:.4f} '
          'Pair_Loss:{pairloss.avg:.4f} '
          'Mse_Loss:{mseloss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           claloss=batch_cla_loss, pairloss=batch_pair_loss, mseloss=batch_mse_loss, acc=corrects))
    

            