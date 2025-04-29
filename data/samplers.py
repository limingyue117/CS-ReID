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

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self,  data_source, model=None, num_instances=4, seed=0, center=None):
        self.data_source = data_source
        self.model = model
        self.center = center
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.seed = seed
        self.epoch = 0
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)

        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length
    
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
    

class RandomIdentitySamplerRAS(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4, seed=0):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.index_1 = defaultdict(list)
        self.index_2 = defaultdict(list)
        self.index_3 = defaultdict(list)
        self.index_4 = defaultdict(list)
        for index, (_, pid, camd, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
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

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            numb = 0
            lenmax = 0
            num1 = len(self.index_1[pid])
            if num1 > 0:
                numb += 1
                lenmax = num1
            num2 = len(self.index_2[pid])
            if num2 > 0:
                numb += 1
                if num2 > lenmax:
                    lenmax = num2
            num3 = len(self.index_3[pid])
            if num3 > 0:
                numb += 1
                if num3 > lenmax:
                    lenmax = num3
            num4 = len(self.index_4[pid])
            if num4 > 0:
                numb += 1
                if num4 > lenmax:
                    lenmax = num4

            num = numb * lenmax
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)
        batch_idxs_dict = defaultdict(list)
        list_container = []
        for pid in self.pids:
            lenmax1 = 0
            biaozhi = 0
            num1 = len(self.index_1[pid])
            num2 = len(self.index_2[pid])
            num3 = len(self.index_3[pid])
            num4 = len(self.index_4[pid])
            list1 = []
            if num1 > 0:
                lenmax1 = num1
                biaozhi = 1
                list1.append(1)
            if num2 > 0:
                list1.append(2)
                if num2 > lenmax1:
                    lenmax1 = num2
                    biaozhi = 2
            if num3 > 0:
                list1.append(3)
                if num3 > lenmax1:
                    lenmax1 = num3
                    biaozhi = 3
            if num4 > 0:
                list1.append(4)
                if num4 > lenmax1:
                    lenmax1 = num4
                    biaozhi = 4
            if len(list1) == 2:
                if biaozhi == 1:
                    idxs0 = copy.deepcopy(self.index_1[pid])
                    list1.remove(1)
                elif biaozhi == 2:
                    idxs0 = copy.deepcopy(self.index_2[pid])
                    list1.remove(2)
                elif biaozhi == 3:
                    idxs0 = copy.deepcopy(self.index_3[pid])
                    list1.remove(3)
                else:
                    idxs0 = copy.deepcopy(self.index_4[pid])
                    list1.remove(4)
                if list1[0] == 1:
                    idxs1 = copy.deepcopy(self.index_1[pid])
                elif list1[0] == 2:
                    idxs1 = copy.deepcopy(self.index_2[pid])
                elif list1[0] == 3:
                    idxs1 = copy.deepcopy(self.index_3[pid])
                else:
                    idxs1 = copy.deepcopy(self.index_4[pid])
                random.shuffle(idxs0)
                random.shuffle(idxs1)
                idxsc = []
                nin1 = 0
                nin0 = 0
                panduan = 0
                while(nin0 < len(idxs0)):
                    if panduan == 0:
                        idxsc.append(idxs0[nin0])
                        nin0 = nin0 +1
                        if nin1 == len(idxs1):
                            nin1 = nin1 - len(idxs1)
                        idxsc.append(idxs1[nin1])
                        nin1 = nin1 +1
                        panduan = 1
                    elif panduan == 1:
                        if nin1 == len(idxs1):
                            nin1 = nin1 - len(idxs1)
                        idxsc.append(idxs1[nin1])
                        nin1 = nin1 +1
                        idxsc.append(idxs0[nin0])
                        nin0 = nin0 +1
                        panduan = 0 
            elif len(list1) == 3:
                if biaozhi == 1:
                    idxs0 = copy.deepcopy(self.index_1[pid])
                    list1.remove(1)
                elif biaozhi == 2:
                    idxs0 = copy.deepcopy(self.index_2[pid])
                    list1.remove(2)
                elif biaozhi == 3:
                    idxs0 = copy.deepcopy(self.index_3[pid])
                    list1.remove(3)
                else:
                    idxs0 = copy.deepcopy(self.index_4[pid])
                    list1.remove(4)
                if list1[0] == 1:
                    idxs1 = copy.deepcopy(self.index_1[pid])
                    list1.remove(1)
                elif list1[0] == 2:
                    idxs1 = copy.deepcopy(self.index_2[pid])
                    list1.remove(2)
                elif list1[0] == 3:
                    idxs1 = copy.deepcopy(self.index_3[pid])
                    list1.remove(3)
                else:
                    idxs1 = copy.deepcopy(self.index_4[pid])
                    list1.remove(4)
                if list1[0] == 1:
                    idxs2 = copy.deepcopy(self.index_1[pid])
                elif list1[0] == 2:
                    idxs2 = copy.deepcopy(self.index_2[pid])
                elif list1[0] == 3:
                    idxs2 = copy.deepcopy(self.index_3[pid])
                else:
                    idxs2 = copy.deepcopy(self.index_4[pid])
                random.shuffle(idxs0)
                random.shuffle(idxs1)
                random.shuffle(idxs2)
                idxsc = []
                nin1 = 0
                nin0 = 0
                nin2 = 0
                while(nin0 < len(idxs0)):
                    idxsc.append(idxs0[nin0])
                    nin0 = nin0 + 1

                    if nin1 == len(idxs1):
                        nin1 = nin1 - len(idxs1)
                    idxsc.append(idxs1[nin1])
                    nin1 = nin1 + 1

                    if nin2 == len(idxs2):
                            nin2 = nin2 - len(idxs2)
                    idxsc.append(idxs2[nin2])
                    nin2 = nin2 + 1
            else:
                if biaozhi == 1:
                    idxs0 = copy.deepcopy(self.index_1[pid])
                    idxs1 = copy.deepcopy(self.index_2[pid])
                    idxs2 = copy.deepcopy(self.index_3[pid])
                    idxs3 = copy.deepcopy(self.index_4[pid])
                elif biaozhi == 2:
                    idxs1 = copy.deepcopy(self.index_1[pid])
                    idxs0 = copy.deepcopy(self.index_2[pid])
                    idxs2 = copy.deepcopy(self.index_3[pid])
                    idxs3 = copy.deepcopy(self.index_4[pid])
                elif biaozhi == 3:
                    idxs1 = copy.deepcopy(self.index_1[pid])
                    idxs2 = copy.deepcopy(self.index_2[pid])
                    idxs0 = copy.deepcopy(self.index_3[pid])
                    idxs3 = copy.deepcopy(self.index_4[pid])
                else:
                    idxs1 = copy.deepcopy(self.index_1[pid])
                    idxs2 = copy.deepcopy(self.index_2[pid])
                    idxs3 = copy.deepcopy(self.index_3[pid])
                    idxs0 = copy.deepcopy(self.index_4[pid])
                random.shuffle(idxs0)
                random.shuffle(idxs1)
                random.shuffle(idxs2)
                random.shuffle(idxs3)
                idxsc = []
                nin1 = 0
                nin0 = 0
                nin2 = 0
                nin3 = 0
                panduan = 0
                while(nin0 < len(idxs0)):
                    if panduan == 0:

                        idxsc.append(idxs0[nin0])
                        nin0 = nin0 +1

                        if nin1 == len(idxs1):
                            nin1 = nin1 - len(idxs1)
                        idxsc.append(idxs1[nin1])
                        nin1 = nin1 +1

                        if nin2 == len(idxs2):
                            nin2 = nin2 - len(idxs2)
                        idxsc.append(idxs2[nin2])
                        nin2 = nin2 +1

                        if nin3 == len(idxs3):
                            nin3 = nin3 - len(idxs3)
                        idxsc.append(idxs3[nin3])
                        nin3 = nin3 +1

                        panduan = 1

                    elif panduan == 1:
                        if nin1 == len(idxs1):
                            nin1 = nin1 - len(idxs1)
                        idxsc.append(idxs1[nin1])
                        nin1 = nin1 +1

                        if nin2 == len(idxs2):
                            nin2 = nin2 - len(idxs2)
                        idxsc.append(idxs2[nin2])
                        nin2 = nin2 +1

                        if nin3 == len(idxs3):
                            nin3 = nin3 - len(idxs3)
                        idxsc.append(idxs3[nin3])
                        nin3 = nin3 +1

                        idxsc.append(idxs0[nin0])
                        nin0 = nin0 +1

                        panduan = 0 
            batch_idxs = []
            for idx in range(len(idxsc)):
                batch_idxs.append(idxsc[idx])
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        avai_pids = copy.deepcopy(self.pids)

        while len(avai_pids) >= 8:
            selected_pids = random.sample(avai_pids, 8) # 随机选取p个id
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0) # 从这个id中选取k张图像
                list_container.append(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:  
                    avai_pids.remove(pid) 

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length
    
    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. This ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


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
    def __init__(self, data_source, model=None, batch_size=64, num_instances=4, transform_train=None, epoch=0, seed=0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.seed = seed
        self.epoch = epoch
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

    def set_model(self, model, epoch):
        """
        Sets the epoch for this sampler. This ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.model = model
        self.epoch = epoch


class GraphSampler_RAS(Sampler):
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
        self.index_1 = defaultdict(list)
        self.index_2 = defaultdict(list)
        self.index_3 = defaultdict(list)
        self.index_4 = defaultdict(list)
        for index, (_, pid, camd, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
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
                remain = [0] * 4
                pid = self.pids[j]                   #每类样本对应原始pid号
                pid_num = len(self.pid_num[pid])
                p1 = 0
                for p in self.pid_num[pid]:
                    if pid_num == 3:
                        if p1 == 0:
                            remain[p] = 3
                        elif p1 == 1:
                            remain[p] = 3
                        else:
                            remain[p] = 2
                        p1 += 1
                    else:
                        remain[p] = int(self.num_instances/pid_num)
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
    
class GraphSampler_Center(Sampler):
    def __init__(self, data_source, model=None, batch_size=64, num_instances=4, transform_train=None, epoch=0, seed=0, center=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.seed = seed
        self.center = center
        self.epoch = epoch
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

    def graph_index(self):
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)
        for pid in self.pids:
            random.shuffle(self.index_dic[pid])
        
        dist = pairwise_distance(self.center, self.center)

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



class GraphSampler_RAS_Center(Sampler):
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
        self.index_1 = defaultdict(list)
        self.index_2 = defaultdict(list)
        self.index_3 = defaultdict(list)
        self.index_4 = defaultdict(list)
        for index, (_, pid, camd, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
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

        self.sam_index = None
        self.sam_pointer1 = [0] * self.num_identities
        self.sam_pointer2 = [0] * self.num_identities
    
    def make_index(self):
        self.graph_index()


    def graph_index(self):
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)
        for pid in self.pids:
            random.shuffle(self.index_dic[pid])
        
        dist = pairwise_distance(self.center, self.center)

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
                img_index1 = self.index_1[pid]      #每个pid对应样本信息
                img_index2 = self.index_2[pid]      #每个pid对应样本信息
                len_p1 = len(img_index1)               #每类样本数目
                len_p2 = len(img_index2)               #每类样本数目
                index_p = []                         #k个样本
                remain1 = int(self.num_instances/2)
                remain2 = int(self.num_instances/2)
                while remain1 > 0:
                    end1 = self.sam_pointer1[j] + remain1              #每类从self.sam_pointer[j]开始，取k个样本
                    idx1 = img_index1[self.sam_pointer1[j] : end1]      
                    index_p.extend(idx1)
                    remain1 -= len(idx1)
                    self.sam_pointer1[j] = end1
                    if end1 >= len_p1:
                        random.shuffle(img_index1)
                        self.sam_pointer1[j] = 0
                while remain2 > 0:
                    end2 = self.sam_pointer2[j] + remain2              #每类从self.sam_pointer[j]开始，取k个样本
                    idx2 = img_index2[self.sam_pointer2[j] : end2]      
                    index_p.extend(idx2)
                    remain2 -= len(idx2)
                    self.sam_pointer2[j] = end2
                    if end2 >= len_p2:
                        random.shuffle(img_index2)
                        self.sam_pointer2[j] = 0
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