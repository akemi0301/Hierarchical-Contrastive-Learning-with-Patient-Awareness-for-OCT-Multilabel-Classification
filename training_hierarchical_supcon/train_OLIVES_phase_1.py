'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import sys
from hierarchical_phase1.data_processing.hierarchical_OLIVES import OLIVES_HierarchihcalDataset, OLIVES_HierarchicalBatchSampler
from hierarchical_phase1.util import adjust_learning_rate, warmup_learning_rate, TwoCropTransform
from hierarchical_phase1.losses import HMLC
from hierarchical_phase1.config_train_OLIVES import parse_option
from hierarchical_phase1.model import LinearClassifier, build_resnet

# from models.query2label_models.backbone import build_backbone
from hierarchical_phase1.model import build_model
import tensorboard_logger as tb_logger
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import shutil
import builtins

best_prec1 = 0
# torch.cuda.empty_cache()
def main():
    global args, best_prec1
    args = parse_option()
    # print(f"args.backbone:{args.backbone}")
    # exit()
    args.save_folder = './save_phase_1/trained_model_hierarchical'
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.tb_folder = './save_phase_1/tensorboard_hierarchical'
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.backbone_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_loss_{}_trial_{}'.\
        format('hmlc', 'dataset', args.backbone, args.learning_rate,
               args.lr_decay_rate, args.batch_size, args.loss, 5)
    if args.tag:
        args.backbone_name = args.backbone_name + '_tag_' + args.tag
    args.tb_folder = os.path.join(args.tb_folder, args.backbone_name)
    args.save_folder = os.path.join(args.save_folder, args.backbone_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # distributed training
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        print("Adopting distributed multi processing training")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    print("GPU in main worker is {}".format(gpu))
    torch.cuda.empty_cache()
    args.gpu = gpu
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print("In the process of multi processing with rank as {}".format(args.rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.backbone))
    model, criterion = set_model(ngpus_per_node, args)
    # print(model.module[0].body.body)
    # exit()
    args.classifier = LinearClassifier(name=args.backbone, num_classes=args.num_classes).cuda(args.gpu)
    set_parameter_requires_grad(model, args)
    optimizer = setup_optimizer(model, args.learning_rate,
                                   args.momentum, args.weight_decay,
                                   args.feature_extract)
    cudnn.benchmark = True
    print(f'args.data:{args.data}')
    # exit()
    # args.data = "data_OLIVES/"
    dataloaders_dict, sampler = load_olives_hierarchical(args.data, args.train_listfile, args.class_map_file,args)

    train_sampler, val_sampler = sampler['train'], sampler['val']
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs + 1))
        print('-' * 50)
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        loss = train(dataloaders_dict, model, criterion, optimizer, epoch, args, logger)
        output_file = args.save_folder + '/checkpoint_loss_{}_epoch_{:04d}_.pth.tar'.format(loss, epoch)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if epoch % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.backbone,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False,
                    filename=output_file)

def set_model(ngpus_per_node, args):
    # model = resnet_modified.MyResNet(name='resnet50')
    if args.backbone == 'resnet50':
        model = build_resnet(args)
    else:
        model = build_model(args)
    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

    # This part is to load a pretrained model
    if args.ckpt != "" and args.backbone == 'resnet50':
    # This part is to load a pretrained model
        ckpt = torch.load(args.ckpt, map_location='cpu')
        # state_dict = ckpt['state_dict']
        state_dict = ckpt['model']
        # state_dict = ckpt
        model_dict = model.state_dict()
        new_state_dict = {}
        # for k, v in state_dict.items():
        #     if not k.startswith('module.encoder_q.fc'):
        #         k = k.replace('module.encoder_q', 'encoder')
        #         new_state_dict[k] = v
        for k, v in state_dict.items():
            if not k.startswith('module.head'):
                k = k.replace('module.encoder', 'encoder')
                new_state_dict[k] = v
        state_dict = new_state_dict
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print("GPU setting", args.gpu)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            print("Updated batch size is {}".format(args.batch_size))
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # There is memory issue in data loader
            # args.workers = 0
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('Loading state dict from ckpt')
            model.load_state_dict(state_dict)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = criterion.cuda(args.gpu)

    return model, criterion

def train(dataloaders, model, criterion, optimizer, epoch, args, logger):
    """
    one epoch training
    """

    classifier = args.classifier
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()

    # Each epoch has a training and/or validation phase
    for phase in ['train']:
        print(len(dataloaders[phase]))
        # exit()
        if phase == 'train':
            # print(phase)
            progress = ProgressMeter(len(dataloaders['train']),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        classifier.eval()

        # Iterate over data.
        for idx, (images, labels) in enumerate(dataloaders[phase]):
            data_time.update(time.time() - end)
            labels = labels.squeeze()
            images = torch.cat([images[0].squeeze(), images[1].squeeze()], dim=0)
            images = images.cuda(non_blocking=True)
            labels = labels.squeeze().cuda(non_blocking=True)
            bsz = labels.shape[0] #batch size
            if phase == 'train':
                warmup_learning_rate(args, epoch, idx, len(dataloaders[phase]), optimizer)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                features = model(images)
                if not isinstance(features, torch.Tensor):
                    features = torch.cat(features[0], dim=0)

                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(features, labels)
                losses.update(loss.item(), bsz)
                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print(idx)
            # print(args.print_freq)
            # exit()

            if idx % args.print_freq == 0:
                progress.display(idx)
            
                sys.stdout.flush()
        logger.log_value('loss', losses.avg, epoch)
    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class TransformRGB(object):
    def __call__(self, img):

        if isinstance(img, torch.Tensor):
            img = transforms.functional.to_pil_image(img)
        img_rgb = img.convert("RGB")


        # Chuyển đổi lại thành tensor PyTorch
        img_rgb_tensor = transforms.functional.to_tensor(img_rgb)

        return img_rgb_tensor

def transform_with_RGB(img_size, normalize):
    transform_rgb = TransformRGB()
    train_transform = transforms.Compose([

            transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1.)),
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

def load_olives_hierarchical(root_dir, train_list_file, class_map_file, opt):
    transform_rgb = TransformRGB()
    mean = (.1706)
    std = (.2112)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0, 0.05), scale=(0.8, 1), shear=10, fill=(255,)),
            transforms.RandomResizedCrop(size=opt.img_size, scale=(0.8, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4)
            ], p=0.8),
            transforms.ToTensor(),
            transform_rgb,
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            normalize
    ])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transform_rgb,
                                    #    transforms.Normalize([0.485, 0.456, 0.406], [
                                    #                         0.229, 0.224, 0.225]),
                                        normalize
                                       ])
    # mean = (.1706)
    # std = (.2112)
    
    # normalize = transforms.Normalize(mean=mean, std=std)
    # print("Going to data ttransform")
    # train_transform = transform_with_RGB(opt.img_size, normalize),
    # val_transform = transform_with_RGB(opt.img_size, normalize)
    
    print(f"root_dir:{root_dir}")
    print(f"os.path.join(root_dir, train_list_file):{os.path.join(root_dir, train_list_file)}")
    # exit()
    train_dataset = OLIVES_HierarchihcalDataset(os.path.join(root_dir, train_list_file),
                                                    os.path.join(root_dir, class_map_file),
                                                    opt,
                                                    transform=TwoCropTransform(train_transform))
    
    # val_dataset = OLIVES_HierarchihcalDataset(os.path.join(root_dir, val_list_file),
    #                                               os.path.join(
    #                                                   root_dir, class_map_file),
    #                                               os.path.join(
    #                                                   root_dir, repeating_product_file),
    #                                               transform=TwoCropTransform(val_transform))
    val_dataset = OLIVES_HierarchihcalDataset(os.path.join(root_dir, train_list_file),
                                                  os.path.join(
                                                      root_dir, class_map_file),
                                                  opt,
                                                  transform=TwoCropTransform(val_transform))
    print('LENGTH TRAIN', len(train_dataset))
    image_datasets = {'train': train_dataset,
                      'val': val_dataset}
    train_sampler = OLIVES_HierarchicalBatchSampler(batch_size=opt.batch_size,
                                       drop_last=False,
                                       dataset=train_dataset)
    val_sampler = OLIVES_HierarchicalBatchSampler(batch_size=opt.batch_size,
                                           drop_last=False,
                                           dataset=val_dataset)
    sampler = {'train': train_sampler,
               'val': val_sampler}
    print(opt.workers, "workers")
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], sampler=sampler[x],
                                       num_workers=opt.workers, batch_size=1,
                                       pin_memory=True)
        for x in ['train', 'val']}
    return dataloaders_dict, sampler


def setup_optimizer(model_ft, lr, momentum, weight_decay, feature_extract):
    # Send the model to GPU
    # model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer_ft

def set_parameter_requires_grad(model, args):
    if args.feature_extract:
        # Select which params to finetune
        # for param in model.parameters():
        #     param.requires_grad = True
        # if args.backbone in ['swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k',  'swin_T_224_22k']:
        #     param_dict = model.named_parammeters()
        # else:
        #     param_dict = model.module.named_parameters()
        for name, param in model.module.named_parameters():
            # print(name)
            if name.startswith('encoder.layer4') or name.startswith('body.layer4'):
                param.requires_grad = True
            elif name.startswith('encoder.layer3') or name.startswith('body.layer3') or name.startswith('layers.3') or name.startswith('layers.2'):
                param.requires_grad = True
            elif name.startswith('head'):
                param.requires_grad = True
            else:
                param.requires_grad = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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



if __name__ == '__main__':
    main()
