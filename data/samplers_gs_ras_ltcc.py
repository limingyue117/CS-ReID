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


class GraphSampler_RAS_LTCC(Sampler):
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
        self.index_5 = defaultdict(list)
        self.index_6 = defaultdict(list)
        self.index_7 = defaultdict(list)
        self.index_8 = defaultdict(list)
        self.index_9 = defaultdict(list)
        self.index_10 = defaultdict(list)
        self.index_11 = defaultdict(list)
        self.index_12 = defaultdict(list)
        self.index_13 = defaultdict(list)
        self.index_14 = defaultdict(list)


        for index, (_, pid, camd, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            self.index_clo[index].append(camd)
            if camd == 0:
                self.index_1[pid].append(index)
            elif camd == 1:
                self.index_2[pid].append(index)
            elif camd == 2:
                self.index_3[pid].append(index)
            elif camd == 3:
                self.index_4[pid].append(index)
            elif camd == 4:
                self.index_5[pid].append(index)
            elif camd == 5:
                self.index_6[pid].append(index)
            elif camd == 6:
                self.index_7[pid].append(index)
            elif camd == 7:
                self.index_8[pid].append(index)
            elif camd == 8:
                self.index_9[pid].append(index)  
            elif camd == 9:
                self.index_10[pid].append(index)
            elif camd == 10:
                self.index_11[pid].append(index)
            elif camd == 11:
                self.index_12[pid].append(index)
            elif camd == 12:
                self.index_13[pid].append(index)
            else:
                self.index_14[pid].append(index)
            
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
            if len(self.index_5[pid])  > 0:
                self.pid_num[pid].append(4)
            if len(self.index_6[pid])  > 0:
                self.pid_num[pid].append(5)
            if len(self.index_7[pid])  > 0:
                self.pid_num[pid].append(6)
            if len(self.index_8[pid])  > 0:
                self.pid_num[pid].append(7)
            if len(self.index_9[pid])  > 0:
                self.pid_num[pid].append(8)
            if len(self.index_10[pid])  > 0:
                self.pid_num[pid].append(9)
            if len(self.index_11[pid])  > 0:
                self.pid_num[pid].append(10)
            if len(self.index_12[pid])  > 0:
                self.pid_num[pid].append(11)
            if len(self.index_13[pid])  > 0:
                self.pid_num[pid].append(12)
            if len(self.index_14[pid])  > 0:
                self.pid_num[pid].append(13)

        self.sam_index = None
        self.sam_pointer1 = [0] * self.num_identities
        self.sam_pointer2 = [0] * self.num_identities
        self.sam_pointer3 = [0] * self.num_identities
        self.sam_pointer4 = [0] * self.num_identities
        self.sam_pointer5 = [0] * self.num_identities
        self.sam_pointer6 = [0] * self.num_identities
        self.sam_pointer7 = [0] * self.num_identities
        self.sam_pointer8 = [0] * self.num_identities
        self.sam_pointer9 = [0] * self.num_identities
        self.sam_pointer10 = [0] * self.num_identities
        self.sam_pointer11 = [0] * self.num_identities
        self.sam_pointer12 = [0] * self.num_identities
        self.sam_pointer13 = [0] * self.num_identities
        self.sam_pointer14 = [0] * self.num_identities
    
    def make_index(self):
        self.graph_index()

    def calc_distance(self, dataset):
        data_loader = DataLoader(
            dataset=ImageDatasetcc(dataset, transform=self.transform_train),
            batch_size=128, num_workers=4,
            shuffle=False, pin_memory=True)
        
        model = deepcopy(self.model).cuda().eval()
        features = extract_features(model, data_loader, 4096)
        dist = pairwise_distance(features, features)

        return dist

    def graph_index(self):
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)
          
        sam_index = []
        pid_index = []
        for pid in self.pids:
            random.shuffle(self.index_dic[pid])
            random.shuffle(self.index_1[pid])
            random.shuffle(self.index_2[pid])
            random.shuffle(self.index_3[pid])
            random.shuffle(self.index_4[pid])
            random.shuffle(self.index_5[pid])
            random.shuffle(self.index_6[pid])
            random.shuffle(self.index_7[pid])
            random.shuffle(self.index_8[pid])
            random.shuffle(self.index_9[pid])
            random.shuffle(self.index_10[pid])
            random.shuffle(self.index_11[pid])
            random.shuffle(self.index_12[pid])
            random.shuffle(self.index_13[pid])
            random.shuffle(self.index_14[pid])
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            cloid = self.index_clo[index][0]
            pid_index.append(cloid)
            sam_index.append(index)

        dataset = [self.data_source[i] for i in sam_index]
        dist = self.calc_distance(dataset)
        # dist = pairwise_distance(self.center, self.center)

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
                remain = [0] * 14
                pid = self.pids[j]                   #每类样本对应原始pid号
                pid_num = self.num_instances
                # pid_sample = random.shuffle(self.pid_num[pid])
                # pid_sample = self.pid_num[pid]
                if len(self.pid_num[pid])>2:
                    pid_zanshi = deepcopy(self.pid_num[pid])
                    gu_pid = pid_index[j]
                    del pid_zanshi[gu_pid]
                    sample = np.random.choice(pid_zanshi, size=1, replace=False)
                    pid_sample = sample.tolist()
                    pid_sample.append(gu_pid)
                    # if pid == 2:
                    #     print(pid_sample)

                else:
                    pid_sample = self.pid_num[pid]

                while pid_num > 0:
                    for p in pid_sample:
                        if pid_num > 0:
                            remain[p] += 1
                            pid_num -= 1

                index_p = []  
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
                while remain[4] > 0:
                    img_index5 = self.index_5[pid]      #每个pid对应样本信息
                    len_p5 = len(img_index5)               #每类样本数目
                    end5 = self.sam_pointer5[j] + remain[4]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx5 = img_index5[self.sam_pointer5[j] : end5]      
                    index_p.extend(idx5)
                    remain[4] -= len(idx5)
                    self.sam_pointer5[j] = end5
                    if end5 >= len_p5:
                        random.shuffle(img_index5)
                        self.sam_pointer5[j] = 0
                while remain[5] > 0:
                    img_index6 = self.index_6[pid]      #每个pid对应样本信息
                    len_p6 = len(img_index6)               #每类样本数目
                    end6 = self.sam_pointer6[j] + remain[5]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx6 = img_index6[self.sam_pointer6[j] : end6]      
                    index_p.extend(idx6)
                    remain[5] -= len(idx6)
                    self.sam_pointer6[j] = end6
                    if end6 >= len_p6:
                        random.shuffle(img_index6)
                        self.sam_pointer6[j] = 0
                while remain[6] > 0:
                    img_index7 = self.index_7[pid]      #每个pid对应样本信息
                    len_p7 = len(img_index7)               #每类样本数目
                    end7 = self.sam_pointer7[j] + remain[6]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx7 = img_index7[self.sam_pointer7[j] : end7]      
                    index_p.extend(idx7)
                    remain[6] -= len(idx7)
                    self.sam_pointer7[j] = end7
                    if end7 >= len_p7:
                        random.shuffle(img_index7)
                        self.sam_pointer7[j] = 0
                while remain[7] > 0:
                    img_index8 = self.index_8[pid]      #每个pid对应样本信息
                    len_p8 = len(img_index8)               #每类样本数目
                    end8 = self.sam_pointer8[j] + remain[7]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx8 = img_index8[self.sam_pointer8[j] : end8]      
                    index_p.extend(idx8)
                    remain[7] -= len(idx8)
                    self.sam_pointer8[j] = end8
                    if end8 >= len_p8:
                        random.shuffle(img_index8)
                        self.sam_pointer8[j] = 0
                while remain[8] > 0:
                    img_index9 = self.index_9[pid]      #每个pid对应样本信息
                    len_p9 = len(img_index9)               #每类样本数目
                    end9 = self.sam_pointer9[j] + remain[8]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx9 = img_index9[self.sam_pointer9[j] : end9]      
                    index_p.extend(idx9)
                    remain[8] -= len(idx9)
                    self.sam_pointer9[j] = end9
                    if end9 >= len_p9:
                        random.shuffle(img_index9)
                        self.sam_pointer9[j] = 0
                while remain[9] > 0:
                    img_index10 = self.index_10[pid]      #每个pid对应样本信息
                    len_p10 = len(img_index10)               #每类样本数目
                    end10 = self.sam_pointer10[j] + remain[9]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx10 = img_index10[self.sam_pointer10[j] : end10]      
                    index_p.extend(idx10)
                    remain[9] -= len(idx10)
                    self.sam_pointer10[j] = end10
                    if end10 >= len_p10:
                        random.shuffle(img_index10)
                        self.sam_pointer10[j] = 0
                while remain[10] > 0:
                    img_index11 = self.index_11[pid]      #每个pid对应样本信息
                    len_p11 = len(img_index11)               #每类样本数目
                    end11 = self.sam_pointer11[j] + remain[10]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx11 = img_index11[self.sam_pointer11[j] : end11]      
                    index_p.extend(idx11)
                    remain[10] -= len(idx11)
                    self.sam_pointer11[j] = end11
                    if end11 >= len_p11:
                        random.shuffle(img_index11)
                        self.sam_pointer11[j] = 0
                while remain[11] > 0:
                    img_index12 = self.index_12[pid]      #每个pid对应样本信息
                    len_p12 = len(img_index12)               #每类样本数目
                    end12 = self.sam_pointer12[j] + remain[11]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx12 = img_index12[self.sam_pointer12[j] : end12]      
                    index_p.extend(idx12)
                    remain[11] -= len(idx12)
                    self.sam_pointer12[j] = end12
                    if end12 >= len_p12:
                        random.shuffle(img_index12)
                        self.sam_pointer12[j] = 0
                while remain[12] > 0:
                    img_index13 = self.index_13[pid]      #每个pid对应样本信息
                    len_p13 = len(img_index13)               #每类样本数目
                    end13 = self.sam_pointer13[j] + remain[12]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx13 = img_index13[self.sam_pointer13[j] : end13]      
                    index_p.extend(idx13)
                    remain[12] -= len(idx13)
                    self.sam_pointer13[j] = end13
                    if end13 >= len_p13:
                        random.shuffle(img_index13)
                        self.sam_pointer13[j] = 0
                while remain[13] > 0:
                    img_index14 = self.index_14[pid]      #每个pid对应样本信息
                    len_p14 = len(img_index14)               #每类样本数目
                    end14 = self.sam_pointer14[j] + remain[13]              #每类从self.sam_pointer[j]开始，取k个样本
                    idx14 = img_index14[self.sam_pointer14[j] : end14]      
                    index_p.extend(idx14)
                    remain[13] -= len(idx14)
                    self.sam_pointer14[j] = end14
                    if end14 >= len_p14:
                        random.shuffle(img_index14)
                        self.sam_pointer14[j] = 0
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
    
    

