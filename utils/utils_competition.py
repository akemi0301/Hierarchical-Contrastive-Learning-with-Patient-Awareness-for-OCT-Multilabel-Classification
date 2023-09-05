from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import os
from sklearn.metrics import roc_auc_score, f1_score
from models.resnet import  SupConResNet,LinearClassifier,LinearClassifier_MultiLabel, SupConResNet_Original, SupConResNet_Original_Headless
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from loss.loss import add_kd_loss


from datasets.biomarker_competition import BiomarkerDataset_Competition
from datasets.biomarker_competition_testing import BiomarkerDataset_Competition_Testing

from models.query2label import Query2Label
from models.query2label_models.transformer import build_transformer

import torch.nn as nn


from models.query2label_models.backbone import build_backbone

def set_model_competition_first(opt):
    backbone = build_backbone(opt) # backbone with pretrained
    # print("Ïn set model competition first")
    # exit()
    params_to_update = backbone.parameters()
    # print("Param to update in competition first:")
    # params_to_update = []
    # for name, param in backbone.named_parameters():
    #     if param.requires_grad == True:
    #         params_to_update.append(param)
    #         print("\t", name)
    # exit()

    transformer_head = build_transformer(opt)

    model = Query2Label(
        backbone = backbone,
        transfomer = transformer_head,
        num_class = opt.num_class
    )

    if not opt.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentity!")

    criterion = torch.nn.BCEWithLogitsLoss()
    device = opt.device
    
    if torch.cuda.is_available():
        if opt.parallel == 0:   
            model = torch.nn.DataParallel(model)

        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

        # backbone.load_state_dict(state_dict)

    return model, criterion

def set_model_competition_second(opt):
    backbone = build_backbone(opt) # backbone with pretrained

    transformer_head = build_transformer(opt)

    # model = Query2Label(
    #     backbone = backbone,
    #     transfomer = transformer_head,
    #     num_class = opt.num_class
    # )

    # if not opt.keep_input_proj:
    #     model.input_proj = nn.Identity()
    #     print("set model.input_proj to Indentity!")

    criterion = torch.nn.BCEWithLogitsLoss()
    device = opt.device
    
    if torch.cuda.is_available():
        if opt.parallel == 0:   
            backbone = torch.nn.DataParallel(backbone)

        backbone = backbone.to(device)
        transformer_head = transformer_head.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

        # backbone.load_state_dict(state_dict)

    return backbone, transformer_head, criterion


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
        


def set_loader_competition(opt):
    mean = (.1706)
    std = (.2112)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.img_size, scale=(0.6, 1.)),
        transforms.RandomHorizontalFlip(),

        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.ToTensor(),
        TransformRGB(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((opt.img_size,opt.img_size)),
        transforms.ToTensor(),
        TransformRGB(),
        normalize,
    ])
    data_path_train = opt.train_image_path
    csv_path_train = opt.train_csv_path

    # data_path_val = opt.val_image_path
    # csv_path_val = opt.val_csv_path

    csv_path_test = opt.test_csv_path
    data_path_test = opt.test_image_path

    train_dataset = BiomarkerDataset_Competition(csv_path_train,data_path_train,transforms = train_transform)
    # if csv_path_val!="":
    #     val_dataset = BiomarkerDataset_Competition(csv_path_val,data_path_val,transforms = val_transform)
    # else:
    #     val_dataset = ""
    test_dataset = BiomarkerDataset_Competition_Testing(csv_path_test,data_path_test,transforms = val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=opt.batch_size, shuffle=True,
    #     num_workers=opt.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return train_loader, test_loader

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
    optimizer = optim.SGD(param_dicts,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)


    return optimizer

# def set_optimizer_query2label(args, model):
#     # optimizer
#     args.lr_mult = args.batch_size / 256
#     if args.optim == 'AdamW':
#         param_dicts = [
#             {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
#         ]
#         optimizer = getattr(torch.optim, args.optim)(
#             param_dicts,
#             args.lr_mult * args.lr,
#             betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
#         )
#     else: 
#         optimizer = None
#     return optimizer

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def accuracy_multilabel(output,target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    r = roc_auc_score(target,output,multi_class='ovr')
    print(r)