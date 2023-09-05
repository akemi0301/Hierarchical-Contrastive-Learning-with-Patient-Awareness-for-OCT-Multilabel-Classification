from torchvision import transforms, datasets

from datasets.biomarker_competition_patient_aware_bcva import PatientAware_BCVA_Dataset

import torch
from models.query2label import build_q2l
from loss.loss import SupConLoss
import torch.backends.cudnn as cudnn
from models.query2label_models.backbone import build_backbone_constrastive
from datasets.prime_trex_combined import CombinedDataset
from utils.utils import TwoCropTransform


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
import torch.nn as nn

def set_model_contrast(opt):

    print(f"train with backbone:{opt.backbone}")
    model = build_backbone_constrastive(opt)

    criterion = SupConLoss(temperature=opt.temp,device=opt.device)
    device = opt.device
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if opt.parallel == 1:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            model = model.to(device)
            criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion

def transform_with_RGB(opt, normalize):
    transform_rgb = TransformRGB()
    train_transform = transforms.Compose([

            transforms.RandomResizedCrop(size=opt.img_size, scale=(0.6, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transform_rgb,
            normalize,
        ])
    return train_transform

def set_loader(opt):
    # construct data loader

    mean = (.1706)
    std = (.2112)
    
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transform_with_RGB(opt, normalize)

    print("Dataset competition")
    csv_path_train = './final_csvs_1/datasets_combined/prime_trex_compressed.csv'
    data_path_train = opt.train_image_path
    train_dataset = PatientAware_BCVA_Dataset(csv_path_train, data_path_train, 
                                              transforms=TwoCropTransform_Patient_Aware(train_transform))
    # train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)

    return train_loader


import cv2
import numpy as np

class TransformRGB(object):
    def __call__(self, img):

        if isinstance(img, torch.Tensor):
            img = transforms.functional.to_pil_image(img)
        img_rgb = img.convert("RGB")


        # Chuyển đổi lại thành tensor PyTorch
        img_rgb_tensor = transforms.functional.to_tensor(img_rgb)

        return img_rgb_tensor

class TwoCropTransform_Patient_Aware:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image_1, image_2):
        return [self.transform(image_1), self.transform(image_2)]