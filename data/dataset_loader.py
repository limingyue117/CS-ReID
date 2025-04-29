import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid
    
class ImageDatasetcc(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id
    
class ImageDatasetSke(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, ske_path, pid, camid, _ = self.dataset[index]
        img = read_image(img_path)
        ske = read_image(ske_path)
        if self.transform is not None:
            img = self.transform(img)
            ske = self.transform(ske)
        return img, ske, pid, camid
    
class ImageDatasetGcnMask(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, height=64, width=32, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.h = height
        self.w = width

    def __len__(self):
        return len(self.dataset)       # 12185

    def __getitem__(self, index):
        try:
            img_path, pid, camid, msk_path = self.dataset[index]
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)       # [3, 256, 128]

            C, H, W = img.shape
            msk = np.load(msk_path)  # [256, 128, 6], min=6.8e-7, max=0.99
            msk = torch.from_numpy(msk).permute(2, 0, 1).unsqueeze(dim=0)  # [1, 6, 256, 128]
            msk = torch.nn.functional.interpolate(msk, size=(self.h, self.w), mode='bilinear', align_corners=True)  # [1, 6, 256, 128], min=8.4-06, max=0.9997

        except:
            print(index)

        return img, msk, pid, camid, img_path