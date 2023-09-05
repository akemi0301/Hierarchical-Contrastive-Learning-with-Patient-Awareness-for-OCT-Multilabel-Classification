from models.query2label_models.tresnet_v2 import TResnetM, TResnetL
from hierarchical_phase1.resnet import ResNet, BasicBlock, Bottleneck
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
from torch.functional import Tensor
from typing import Type, Any, Union, List
from models.query2label_models.swin_transformer import build_swin_transformer
from typing import Dict, List
__all__ = ['ResNet', 'resnet50', 'tresnetm_v2', 'tresnetl_v2', 'swin_T_224_22k']

import torch
torch.cuda.empty_cache()

model_urls = {
    # 'resnet50': './pretrained_imagenet_dir/resnet50-11ad3fa6.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    'tresnetm_v2':'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m_448.pth',
    'tresnetl_v2':'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_l_448.pth',
    'swin_B_224_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
    'swin_B_384_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
    'swin_L_224_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
    'swin_L_384_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
    'swin_T_224_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth'
}


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print("Using pretrained for resnet50!")
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

model_dict = {
    'resnet50': [resnet50, 2048],
    'tresnetm_v2': ['tresnetm_v2', 2048],
    'tresnetl_v2': ['tresnetl_v2',2048],
    'swin_B_224_22k': ['swin_B_224_22k',1024],
    'swin_B_384_22k': ['swin_B_384_22k',1024],
    'swin_L_224_22k': ['swin_L_224_22k',1024],
    'swin_L_384_22k': ['swin_L_384_22k',1024],
    'swin_T_224_22k': ['swin_T_224_22k',768],    
}

class build_resnet(nn.Module):
    """backbone + projection head"""
    def __init__(self,args,  name='resnet50', head='mlp', feat_dim=128):
        super(build_resnet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(pretrained = args.pretrained)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

def build_model(args):
    head = 'mlp'
    feat_dim = 128
    _, dim_in = model_dict[args.backbone]
    # position_embedding = build_position_encoding(args)
    train_backbone = True

    if args.backbone == 'resnet50':
       model = build_resnet(args)
       return model
    
    elif args.backbone in ['tresnetl_v2', 'tresnetm_v2']:
        if args.backbone == 'tresnetm_v2':
            model = TResnetM(feat_dim)
        else:
            # return_interm_layers = False
            model = TResnetL(feat_dim)
    elif args.backbone in ['swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k',  'swin_T_224_22k']:
        imgsize = int(args.backbone.split('_')[-2])
        model = build_swin_transformer(args.backbone, imgsize)
    
    if args.pretrained and args.ckpt == '':
            print('use pretrained: True')
            state_dict = load_state_dict_from_url(model_urls[args.backbone],
                                                    progress=True)
            model.load_state_dict(state_dict, strict=False)

    if head == 'linear':
        model.head = nn.Linear(dim_in, feat_dim)
    elif head == 'mlp':
        model.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
    else:
        raise NotImplementedError('head not supported: {}'.format(head))
    model.forward = model.forward_SupCon
    # print(model)
    # exit()

    return model

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        
        for name, param in model.named_parameters():
            # print(name)
            if name.startswith('encoder.layer4') or name.startswith('body.layer4'):
                param.requires_grad = True
            elif name.startswith('encoder.layer3') or name.startswith('body.layer3') or name.startswith('layers.3'):
                param.requires_grad = True
            elif name.startswith('head'):
                param.requires_grad = True
            else:
                param.requires_grad = False