from torchvision import transforms, datasets

from datasets.oct_dataset import OCTDataset
from datasets.biomarker import BiomarkerDatasetAttributes
from utils.utils import TwoCropTransform
from datasets.prime import PrimeDatasetAttributes
from datasets.prime_trex_combined import CombinedDataset
from datasets.recovery import recovery
from datasets.trex import TREX
import torch
from models.resnet import SupConResNet, SupConResNet_Original
from models.query2label import build_q2l
from loss.loss import SupConLoss
import torch.backends.cudnn as cudnn
from models.query2label_models.backbone import build_backbone
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
import torch.nn as nn
def set_model_contrast(opt):

    if opt.model is not None:
        print(f"ïs model none:{opt.model}")
        model = SupConResNet_Original(name=opt.model)
    else:
        print(f"train with backbone:{opt.backbone}")
        model = build_backbone(opt)

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


    if opt.dataset == 'OCT':

        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'OCT_Cluster':

        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'Prime' or opt.dataset == 'CombinedBio' or opt.dataset == 'CombinedBio_Modfied' or opt.dataset =='Prime_Compressed':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'TREX_DME' or opt.dataset == 'Prime_TREX_DME_Fixed' \
            or opt.dataset == 'Prime_TREX_Alpha' or opt.dataset == 'Prime_TREX_DME_Discrete' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX'\
            or opt.dataset == "Competition":
        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'PrimeBio':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'Prime_Comb_Bio':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'Recovery' or opt.dataset == 'Recovery_Compressed':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'path':

        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    
    if "swin" in opt.backbone:
        train_transform = transform_with_RGB(opt, normalize)
    elif "CvT" in opt.backbone:
        train_transform = transform_with_RGB(opt, normalize)
    
    else:

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])


    print(f"opt.dataset:{opt.dataset}")
    if opt.dataset =='OCT':
        csv_path_train = opt.train_csv_path
        data_path_train = opt.train_image_path
        train_dataset = OCTDataset(csv_path_train,data_path_train,transforms = TwoCropTransform(train_transform))
    elif opt.dataset == 'CombinedBio':
        csv_path_train = '/home/kiran/Desktop/Dev/SupCon_OCT_Clinical/final_csvs_1/biomarker_csv_files/complete_biomarker_training.csv'
        data_path_train = opt.train_image_path
        train_dataset = BiomarkerDatasetAttributes(csv_path_train,data_path_train,transforms = TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime':
        csv_path_train = './final_csvs_' + str(opt.patient_split) + '/complete_prime_recovery_trex'+'/full_prime_train.csv'
        data_path_train = opt.train_image_path
        train_dataset = PrimeDatasetAttributes(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))

    elif opt.dataset == 'Prime_TREX_DME_Fixed' or opt.dataset == 'Prime_TREX_Alpha' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/datasets_combined/prime_trex_compressed.csv'
        data_path_train = opt.train_image_path
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == "Competition":
        print("Dataset competition")
        csv_path_train = './final_competition_csv/Training_Unlabeled_Clinical_Data.csv'
        data_path_train = opt.train_image_path
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)

    return train_loader


def set_model(opt):

    model = SupConResNet_Original(name=opt.model)

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

import cv2
import numpy as np

class TransformRGB(object):
    def __call__(self, img):
        # Chuyển đổi mảng numpy thành tensor PyTorch
        # img_tensor = transforms.functional.to_tensor(img)

        # Chuyển đổi tensor từ kênh BGR sang kênh RGB
        # img_rgb = cv2.cvtColor(img_tensor.numpy(), cv2.COLOR_BGR2RGB)
        if isinstance(img, torch.Tensor):
            img = transforms.functional.to_pil_image(img)
        img_rgb = img.convert("RGB")


        # Chuyển đổi lại thành tensor PyTorch
        img_rgb_tensor = transforms.functional.to_tensor(img_rgb)

        return img_rgb_tensor