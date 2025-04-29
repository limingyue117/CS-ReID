import data.transforms as T
from torch.utils.data import DataLoader
from data.datasets import Market1501, CUHK03, DukeMTMCreID, MSMT17, CELEB_REID, PRCC, NKUP
from data.dataset_loader import ImageDataset, ImageDatasetcc, ImageDatasetGcnMask, ImageDatasetSke
from data.samplers import RandomIdentitySampler, RandomIdentitySamplerRAS
from data.samplers_gs import GraphSampler
from data.samplers_gs_ras import GraphSampler_RAS
from data.samplers_gs_center import GraphSampler_Center
from data.samplers_gs_ras_center import GraphSampler_RAS_Center
from data.samplers_gs_ras_prcc import GraphSampler_RAS_PRCC
from data.samplers_gs_ras_nkup import GraphSampler_RAS_NKUP
from data.samplers_gs_ras2_nkup import GraphSampler_RAS2_NKUP
from data.samplers_gs_ras_ltcc import GraphSampler_RAS_LTCC
from data.samplers_gs_bas_ltcc import GraphSampler_BAS_LTCC
from data.prcc_gcn import PRCC_GCN
from data.ltcc import LTCC
from data.ltcc_ori import LTCC_ORI
from data.prcc_ske import PRCC_SKE
import torch

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'celeb_reid':CELEB_REID,
    'prcc':PRCC,
    'prcc_cal':PRCC,
    'prcc_ske':PRCC_SKE,
    'nkup':NKUP,
    'prcc_gcn':PRCC_GCN,
    'ltcc':LTCC,
    'ltcc_ori':LTCC_ORI
}


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))

    print("Initializing dataset {}".format(config.DATA.DATASET))
    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, split_id=config.DATA.SPLIT_ID,
                                             cuhk03_labeled=config.DATA.CUHK03_LABELED, 
                                             cuhk03_classic_split=config.DATA.CUHK03_CLASSIC_SPLIT)

    return dataset


def build_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test

def train_collate_gcn_mask(batch):
    imgs, masks, pids, _, pathes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs), pids, pathes, torch.cat(masks)

def build_dataloader(config):
    dataset = build_dataset(config)
    transform_train, transform_test = build_transforms(config)
    if config.DATA.DATASET == 'prcc' or config.DATA.DATASET == 'prcc_cal':
        # train_sampler = RandomIdentitySampler(dataset.train, 
        #                                                  num_instances=config.DATA.NUM_INSTANCES,
        #                                                  seed=config.SEED)
        train_sampler = GraphSampler_RAS(dataset.train, batch_size=64, num_instances=config.DATA.NUM_INSTANCES, transform_train=transform_train, seed=config.SEED, dim_feature=config.MODEL.FEATURE_DIM)
        trainloader = DataLoader(dataset=ImageDatasetcc(dataset.train, transform=transform_train),
                                    sampler=train_sampler,
                                    batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=True)
        queryloader_same = DataLoader(dataset=ImageDatasetcc(dataset.query_same, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        queryloader_diff = DataLoader(dataset=ImageDatasetcc(dataset.query_diff, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        galleryloader = DataLoader(dataset=ImageDatasetcc(dataset.gallery, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        
        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler
    

    elif config.DATA.DATASET == 'ltcc' or config.DATA.DATASET == 'ltcc_ori':
        # train_sampler = RandomIdentitySampler(dataset.train, 
        #                                                  num_instances=config.DATA.NUM_INSTANCES,
        #                                                  seed=config.SEED)
        train_sampler = GraphSampler_RAS_LTCC(dataset.train, batch_size=config.DATA.TRAIN_BATCH, 
                                              num_instances=config.DATA.NUM_INSTANCES, 
                                              transform_train=transform_train, seed=config.SEED)
        trainloader = DataLoader(dataset=ImageDatasetcc(dataset.train, transform=transform_train),
                                    sampler=train_sampler,
                                    batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=True)
        queryloader = DataLoader(dataset=ImageDatasetcc(dataset.query, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        # queryloader_diff = DataLoader(dataset=ImageDatasetcc(dataset.query_diff, transform=transform_test),
        #                             batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
        #                             pin_memory=True, drop_last=False, shuffle=False)
        galleryloader = DataLoader(dataset=ImageDatasetcc(dataset.gallery, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        
        return trainloader, queryloader, galleryloader, dataset, train_sampler
    
    elif config.DATA.DATASET == 'nkup':
        train_sampler = GraphSampler_RAS2_NKUP(dataset.train, batch_size=96, num_instances=config.DATA.NUM_INSTANCES, transform_train=transform_train, seed=config.SEED)
        trainloader = DataLoader(dataset=ImageDatasetcc(dataset.train, transform=transform_train),
                                    sampler=train_sampler,
                                    batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=True)
        queryloader = DataLoader(dataset=ImageDatasetcc(dataset.query, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        galleryloader_same = DataLoader(dataset=ImageDatasetcc(dataset.gallery_same, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        galleryloader_diff = DataLoader(dataset=ImageDatasetcc(dataset.gallery_diff, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        galleryloader_ldiff = DataLoader(dataset=ImageDatasetcc(dataset.gallery_ldiff, transform=transform_test),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        galleryloader_diff_all = DataLoader(dataset=ImageDatasetcc(dataset.gallery_diff_all, transform=transform_test),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)
        return trainloader, queryloader, galleryloader_same, galleryloader_diff, galleryloader_ldiff, galleryloader_diff_all, dataset, train_sampler
    
    else:
        trainloader = DataLoader(ImageDataset(dataset.train, transform=transform_train),
                                sampler=RandomIdentitySampler(dataset.train, num_instances=config.DATA.NUM_INSTANCES),
                                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True)
        queryloader = DataLoader(ImageDataset(dataset.query, transform=transform_test),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=False, shuffle=False)
        galleryloader = DataLoader(ImageDataset(dataset.gallery, transform=transform_test),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=False, shuffle=False)
        
        return trainloader, queryloader, galleryloader, dataset.num_train_pids
