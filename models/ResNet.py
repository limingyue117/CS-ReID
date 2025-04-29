import torchvision
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from models.utils import pooling


class ResNet50(nn.Module):
    def __init__(self, res4_stride=1, modelname ='', dim_feature = 2048, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if res4_stride == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        

        if modelname == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif modelname == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif modelname == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(modelname))
        

        self.bn = nn.BatchNorm1d(dim_feature)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, x):
        x = self.base(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f