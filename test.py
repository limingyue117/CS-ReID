import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
import random


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)

    return img_flip


@torch.no_grad()
def extract_feature(model, dataloader):
    features, pids, camids, clothes_ids = [], [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs).data.cpu()

        batch_features_flip = model(flip_imgs).data.cpu()

        batch_features += batch_features_flip
        
        batch_features = F.normalize(batch_features, p=2, dim=-1)

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
        clothes_ids.append(batch_clothes_ids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()
    clothes_ids =  torch.cat(clothes_ids, 0).numpy()

    return features, pids, camids, clothes_ids

@torch.no_grad()
def extract_ske_feature(model, dataloader):
    features, features_ori, features_ske, pids, camids = [], [], [], [], []
    for batch_idx, (imgs, skes, batch_pids, batch_camids) in enumerate(dataloader):
        # flip_imgs = fliplr(imgs)
        # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        imgs = imgs.cuda()
        features1 = model(imgs).data.cpu()
        # features1 += model(flip_imgs).data.cpu()

        # flip_skes = fliplr(skes)
        # skes, flip_skes = skes.cuda(), flip_skes.cuda()
        skes = skes.cuda()
        features2 = model(skes).data.cpu()
        # features2 += model(flip_skes).data.cpu()

        batch_features = torch.cat([features1,features2],1)

        features.append(batch_features)
        features_ori.append(features1)
        features_ske.append(features2)
        pids.append(batch_pids)
        camids.append(batch_camids)
    features = torch.cat(features, 0)
    features_ori = torch.cat(features_ori, 0)
    features_ske = torch.cat(features_ske, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()

    return features, features_ori, features_ske, pids, camids

def euclidean_distance(config, qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    distmat = torch.zeros((m,n))
    if config.TEST.DISTANCE == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        # Cosine similarity
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    return distmat


def test(config, model, queryloader, galleryloader, dataset):
    since = time.time()
    model.eval()
    # Extract features 
    qf, q_pids, q_camids, q_clothes_ids = extract_feature(model, queryloader)
    gf, g_pids, g_camids, g_clothes_ids = extract_feature(model, galleryloader)
    # Gather samples from different GPUs

    time_elapsed = time.time() - since
    
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()

    time_elapsed = time.time() - since
    print('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    print('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")

    print("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")

    return cmc[0]
    


def test_prcc(config, model, queryloader_same, queryloader, galleryloader, dataset):
    since = time.time()
    model.eval()
    
    # Extract features for query set
    qf, q_pids, q_camids, _ = extract_feature(model, queryloader)
    # Extract features for query set
    qsf, qs_pids, qs_camids, _ = extract_feature(model, queryloader_same)
    # print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # Extract features for gallery set
    gf, g_pids, g_camids, _ = extract_feature(model, galleryloader)
    # print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # Compute distance matrix between query and gallery
    distmat0 = euclidean_distance(config, qsf, gf)

    print("Computing CMC and mAP same")
    cmc0, mAP0 = evaluate(distmat0, qs_pids, g_pids, qs_camids, g_camids)

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc0[0], cmc0[4], cmc0[9], mAP0))
    print("------------------------------------------------")
    
    # Compute distance matrix between query and gallery
    distmat = euclidean_distance(config, qf, gf)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")


    feat_idx = {}
    pid_idx = {}
    camid_idx = {}
    distmat_idx = {}
    for i in range(0,10):
        feat_idx[i] = []
        pid_idx[i] = []
        camid_idx[i] = []
        distmat_idx[i] = []
        for j in dataset.gallery_idx[i]:
            feat_idx[i].append(gf[j])
            pid_idx[i].append(g_pids[j])
            camid_idx[i].append(g_camids[j])
        distmat_idx[i] = euclidean_distance(config, qf, torch.stack(feat_idx[i]))
        if i == 0:
            cmc1, mAP1 = evaluate(distmat_idx[i], q_pids, np.asarray(pid_idx[i]), q_camids, np.asarray(camid_idx[i]))
        else:
            cmc1_zanshi, mAP1_zanshi = evaluate(distmat_idx[i], q_pids, np.asarray(pid_idx[i]), q_camids, np.asarray(camid_idx[i]))
            cmc1 += cmc1_zanshi
            mAP1 += mAP1_zanshi
    cmc1 = cmc1/10.0
    mAP1 = mAP1/10.0

    print("Computing CMC and mAP ori")
    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc1[0], cmc1[4], cmc1[9], mAP1))
    print("------------------------------------------------")

    return cmc[0]


def test_ltcc(config, model, queryloader, galleryloader, dataset):
    since = time.time()
    model.eval()
    
    # Extract features for query set
    qf, q_pids, q_camids, _ = extract_feature(model, queryloader)
    # Extract features for query set
    # qsf, qs_pids, qs_camids = extract_feature(model, queryloader_same)
    # print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # Extract features for gallery set
    gf, g_pids, g_camids, _ = extract_feature(model, galleryloader)
    # print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # Compute distance matrix between query and gallery
    # distmat0 = euclidean_distance(config, qsf, gf)

    # print("Computing CMC and mAP same")
    # cmc0, mAP0 = evaluate(distmat0, qs_pids, g_pids, qs_camids, g_camids)

    # print("Results ----------------------------------------")
    # print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc0[0], cmc0[4], cmc0[9], mAP0))
    # print("------------------------------------------------")
    
    # Compute distance matrix between query and gallery
    distmat = euclidean_distance(config, qf, gf)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")


    return cmc[0]


def test_nkup(config, model, queryloader, galleryloader_same, galleryloader_diff, galleryloader_ldiff, galleryloader_diff_all, dataset):
    since = time.time()
    model.eval()
    # Extract features for query set
    gsf, gs_pids, gs_camids, _ = extract_feature(model, galleryloader_same)
    gdf, gd_pids, gd_camids, _ = extract_feature(model, galleryloader_diff)
    gdlf, gdl_pids, gdl_camids, _ = extract_feature(model, galleryloader_ldiff)
    gdf_all, gd_all_pids, gd_all_camids, _ = extract_feature(model, galleryloader_diff_all)
    # Extract features for gallery set
    qf, q_pids, q_camids, _ = extract_feature(model, queryloader)
    time_elapsed = time.time() - since
    
    print("Extracted features for gallery set (with same clothes), obtained {} matrix".format(gsf.shape))
    print("Extracted features for gallery set (with different clothes), obtained {} matrix".format(gdf.shape))
    print("Extracted features for gallery set (with large different clothes), obtained {} matrix".format(gdlf.shape))
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    # Cosine similarity
    distmat_same = euclidean_distance(config, qf, gsf)
    distmat_diff = euclidean_distance(config, qf, gdf)
    distmat_ldiff = euclidean_distance(config, qf, gdlf)
    distmat_all = euclidean_distance(config, qf, gdf_all)


    print("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, q_pids, gs_pids, q_camids, gs_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")


    print("Computing CMC and mAP only for clothes changing")
    cmc1, mAP1 = evaluate(distmat_diff, q_pids, gd_pids, q_camids, gd_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1))
    print("-----------------------------------------------------------")


    print("Computing CMC and mAP only for large clothes changing")
    cmc2, mAP2 = evaluate(distmat_ldiff, q_pids, gdl_pids, q_camids, gdl_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2))
    print("-----------------------------------------------------------")

    print("Computing CMC and mAP only for all clothes changing")
    cmc3, mAP3 = evaluate(distmat_all, q_pids, gd_all_pids, q_camids, gd_all_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc3[0], cmc3[4], cmc3[9], cmc3[19], mAP3))
    print("-----------------------------------------------------------")


    return cmc3[0]