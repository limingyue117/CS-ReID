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


class GraphSampler_RAS2_NKUP(Sampler):
    def __init__(self, data_source, model=None, batch_size=96, num_instances=4, transform_train=None, epoch=0, seed=0, center=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.seed = seed
        self.epoch = epoch
        self.center = center
        self.transform_train = transform_train
        self.index_dic = defaultdict(list)
        self.index_clo = defaultdict(list)
        self.index_1 = defaultdict(list)
        self.index_2 = defaultdict(list)
        self.index_3 = defaultdict(list)
        self.index_4 = defaultdict(list)
        for index, (_, pid, camd, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            self.index_clo[index].append(camd)
            if camd == 0:
                self.index_1[pid].append(index)
            elif camd == 1:
                self.index_2[pid].append(index)
            elif camd == 2:
                self.index_3[pid].append(index)
            else:
                self.index_4[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self.pid_num = defaultdict(list)
        for pid in self.pids:
            if len(self.index_1[pid]) > 0:
                self.pid_num[pid].append(0)
            if len(self.index_2[pid])  > 0:
                self.pid_num[pid].append(1)
            if len(self.index_3[pid]) > 0:
                self.pid_num[pid].append(2)
            if len(self.index_4[pid])  > 0:
                self.pid_num[pid].append(3)
    
        self.sam_index = None
        self.sam_pointer1 = [0] * self.num_identities
        self.sam_pointer2 = [0] * self.num_identities
        self.sam_pointer3 = [0] * self.num_identities
        self.sam_pointer4 = [0] * self.num_identities
    
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
            random.shuffle(self.index_1[pid])
            random.shuffle(self.index_2[pid])
            random.shuffle(self.index_3[pid])
            random.shuffle(self.index_4[pid])
            
        sam_index = []
        pid_index = []
        for pid in self.pids:
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            cloid = self.index_clo[index][0]
            pid_index.append(cloid)
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
                remain = [0] * 4
                pid = self.pids[j]                   #每类样本对应原始pid号
                pid_num = self.num_instances

                if len(self.pid_num[pid])>2:
                    pid_zanshi = deepcopy(self.pid_num[pid])
                    gu_pid = pid_index[j]
                    del pid_zanshi[gu_pid]
                    sample = np.random.choice(pid_zanshi, size=1, replace=False)
                    pid_sample = sample.tolist()
                    pid_sample.append(gu_pid)
                else:
                    pid_sample = self.pid_num[pid]

                while pid_num > 0:
                    for p in pid_sample:
                        if pid_num > 0:
                            remain[p] += 1
                            pid_num -= 1

                index_p = []                         #k个样本
                while remain[0] > 0:
                    img_index1 = self.index_1[pid]      #每个pid对应样本信息
                    len_p1 = len(img_index1)               #每类样本数目
                    end1 = self.sam_pointer1[j] + remain[0]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx1 = img_index1[self.sam_pointer1[j] : end1]      
                    index_p.extend(idx1)
                    remain[0] -= len(idx1)
                    self.sam_pointer1[j] = end1
                    if end1 >= len_p1:
                        random.shuffle(img_index1)
                        self.sam_pointer1[j] = 0
                while remain[1] > 0:
                    img_index2 = self.index_2[pid]      #每个pid对应样本信息
                    len_p2 = len(img_index2)               #每类样本数目
                    end2 = self.sam_pointer2[j] + remain[1]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx2 = img_index2[self.sam_pointer2[j] : end2]      
                    index_p.extend(idx2)
                    remain[1] -= len(idx2)
                    self.sam_pointer2[j] = end2
                    if end2 >= len_p2:
                        random.shuffle(img_index2)
                        self.sam_pointer2[j] = 0
                while remain[2] > 0:
                    img_index3 = self.index_3[pid]      #每个pid对应样本信息
                    len_p3 = len(img_index3)               #每类样本数目
                    end3 = self.sam_pointer3[j] + remain[2]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx3 = img_index3[self.sam_pointer3[j] : end3]      
                    index_p.extend(idx3)
                    remain[2] -= len(idx3)
                    self.sam_pointer3[j] = end3
                    if end3 >= len_p3:
                        random.shuffle(img_index3)
                        self.sam_pointer3[j] = 0
                while remain[3] > 0:
                    img_index4 = self.index_4[pid]      #每个pid对应样本信息
                    len_p4 = len(img_index4)               #每类样本数目
                    end4 = self.sam_pointer4[j] + remain[3]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx4 = img_index4[self.sam_pointer4[j] : end4]      
                    index_p.extend(idx4)
                    remain[3] -= len(idx4)
                    self.sam_pointer4[j] = end4
                    if end4 >= len_p4:
                        random.shuffle(img_index4)
                        self.sam_pointer4[j] = 0
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
    
    

