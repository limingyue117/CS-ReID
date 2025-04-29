import copy
import random
import time
import torch
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from data.dataset_loader import ImageDataset, ImageDatasetcc
import data.transforms as T

def extract_cnn_feature(model, inputs):
    inputs = inputs.cuda()
    outputs = model(inputs)
    return outputs

def extract_features(model, data_loader, num_features):

    shape = [len(data_loader.dataset), num_features]
    features = torch.zeros(shape)

    for i, (imgs, _, _, _) in enumerate(data_loader):
        k0 = i * data_loader.batch_size
        k1 = k0 + len(imgs)
        features[k0 : k1, :] = extract_cnn_feature(model, imgs)
    return features


def pairwise_distance(prob_fea, gal_fea):
    m = gal_fea.size(0)
    n = prob_fea.size(0)
    distmat = torch.zeros((m,n))

    # Cosine similarity
    qf = F.normalize(prob_fea, p=2, dim=1)
    gf = F.normalize(gal_fea, p=2, dim=1)
    for i in range(m):
        distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    return distmat

class GraphSampler(Sampler):
    def __init__(self, data_source, model=None, batch_size=64, num_instances=4, transform_train=None, epoch=0, seed=0, center=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.seed = seed
        self.epoch = epoch
        self.center = center
        self.transform_train = transform_train
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self.sam_index = None
        self.sam_pointer = [0] * self.num_identities
    
    def make_index(self):
        self.graph_index()

    def calc_distance(self, dataset):
        data_loader = DataLoader(
            dataset=ImageDatasetcc(dataset, transform=self.transform_train),
            batch_size=256, num_workers=8,
            shuffle=False, pin_memory=True)

        model = deepcopy(self.model).cuda().eval()
        features = extract_features(model, data_loader, 4096)
        dist = pairwise_distance(features, features)
        return dist

    def graph_index(self):
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)
        for pid in self.pids:
            random.shuffle(self.index_dic[pid])
        sam_index = []
        for pid in self.pids:
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            sam_index.append(index)

        dataset = [self.data_source[i] for i in sam_index]
        dist = self.calc_distance(dataset)

        with torch.no_grad():
            dist = dist + torch.eye(self.num_identities, device=dist.device) * 1e15
            topk = self.batch_size // self.num_instances - 1
            _, topk_index = torch.topk(dist.cuda(), topk, largest=False)
            topk_index = topk_index.cpu().numpy()  

        sam_index = []
        for i in range(self.num_identities):
            id_index = topk_index[i, :].tolist()     #与第i类相邻的p-1类
            id_index.append(i)                       #加上第i类构建p类
            index = []
            for j in id_index:                       #每类取k个样本
                pid = self.pids[j]                   #每类样本对应原始pid号
                img_index = self.index_dic[pid]      #每个pid对应样本信息
                len_p = len(img_index)               #每类样本数目
                index_p = []                         #k个样本
                remain = self.num_instances
                while remain > 0:
                    end = self.sam_pointer[j] + remain              #每类从self.sam_pointer[j]开始，取k个样本
                    idx = img_index[self.sam_pointer[j] : end]
                    index_p.extend(idx)
                    remain -= len(idx)
                    self.sam_pointer[j] = end
                    if end >= len_p:
                        random.shuffle(img_index)
                        self.sam_pointer[j] = 0
                assert(len(index_p) == self.num_instances)
                index.extend(index_p)
            sam_index.extend(index)

        sam_index = np.array(sam_index)
        sam_index = sam_index.reshape((-1, self.batch_size))
        np.random.shuffle(sam_index)
        sam_index = list(sam_index.flatten())
        self.sam_index = sam_index 

    def __iter__(self):
        self.make_index()
        return iter(self.sam_index)

    def __len__(self):
        if self.sam_index is None:
            return self.num_identities
        else:
            return len(self.sam_index)

    def set_model(self, model, epoch, center):
        """
        Sets the epoch for this sampler. This ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.model = model  
        self.epoch = epoch 
        self.center = center.detach()
    

