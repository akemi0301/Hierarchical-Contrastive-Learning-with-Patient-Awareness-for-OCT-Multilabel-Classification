import torch
from torch.nn import ModuleList
from utils.utils import AverageMeter,warmup_learning_rate, accuracy, save_model
import sys
import time
import numpy as np
from config.config_linear_competition import parse_option
from utils.utils_competition import set_loader_competition, set_model_competition_first, set_optimizer, adjust_learning_rate, accuracy_multilabel
from sklearn.metrics import average_precision_score,roc_auc_score, classification_report
import pandas as pd
from visualization.grad_cam import visualize

import os

def train_OCT_multilabel(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    # model.eval()
    # classifier.train()
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()
 
    # swin_norm = [model.backbone[0].layers[-1].blocks[-1].norm2]
    # trans_encoder_norm = [model.transformer.encoder.layers[-1].norm2]
    # trans_decoder1_norm = [model.transformer.decoder.layers[0].norm2]
    # trans_decoder2_norm = [model.transformer.decoder.layers[1].norm2]

    # print(len(model.backbone))
    # print(swin_norm)
    # print(model.transformer.encoder)
    # print('======================================================================')

    for idx, (image, bio_tensor, img_paths) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = image.to(device)

        labels = bio_tensor
        labels = labels.float()
        bsz = labels.shape[0]
        labels=labels.to(device)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)


        with torch.cuda.amp.autocast(enabled=opt.amp):
            output = model(images)
            loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t loss:{3}'.format(
                epoch, idx + 1, len(train_loader), loss))
            # print(img_paths)
            # i = 1
            # if i == 1:
            #     visualize(opt.grad_method, model, swin_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
            #     i += 1
            # if i == 2:
            #     visualize(opt.grad_method, model, trans_encoder_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
            #     i += 1
            # if i == 3:
            #     visualize(opt.grad_method, model, trans_decoder1_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
            #     i += 1
            # if i == 4:
            #     visualize(opt.grad_method, model, trans_decoder2_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
            
            sys.stdout.flush()
        
        if idx == int(len(train_loader)/2):
            if opt.grad_visualize: 
                i = 1
                if i == 1:
                    visualize(opt.grad_method, model, swin_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
                    i += 1
                if i == 2:
                    visualize(opt.grad_method, model, trans_encoder_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
                    i += 1
                
        if idx == len(train_loader)-1:
            if opt.grad_visualize: 
                i = 1
                if i == 1:
                    visualize(opt.grad_method, model, swin_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
                    i += 1
                if i == 2:
                    visualize(opt.grad_method, model, trans_encoder_norm, img_paths, epoch, idx+1, opt.img_visualize, i)
                    i += 1

        
    return losses.avg, top1.avg

def validate_multilabel(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    # classifier.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, bio_tensor) in enumerate(val_loader):
            images = image.float().to(device)

            labels = bio_tensor
            labels = labels.float()
            print(idx)
            label_list.append(labels.squeeze().detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            # output = classifier(model.encoder(images))

            loss = criterion(output, labels)
            output = torch.round(torch.sigmoid(output))

            # compute output
            with torch.cuda.amp.autocast(enabled=opt.amp):
                output = model(images)
                loss = criterion(output, labels)
                output = torch.round(torch.sigmoid(output))

            out_list.append(output.squeeze().detach().cpu().numpy())
            # update metric
            # losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    label_array = np.array(label_list)
    out_array = np.array(out_list)
    out_array = np.concatenate(out_list, axis=0)
    r = roc_auc_score(label_array, out_array, average='macro')


    return losses.avg, r



def test_multilabel(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    # classifier.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, img_name) in enumerate(val_loader):
            images = image.float().to(device)

            # print(f"idx:{idx}")

            # compute output
            with torch.cuda.amp.autocast(enabled=opt.amp):
                output = model(images)
                output = torch.round(torch.sigmoid(output))

            
            output = output.squeeze().detach().cpu().numpy().tolist()
            row = img_name + output
            # print(f"row:{row}")
            out_list.append(row)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # out_array = np.array(out_list)
    # out_array = np.concatenate(out_list, axis=0)

    return out_list

def main_multilabel_competition():
    best_acc = 0
    opt = parse_option()

    # build data loader
    device = opt.device
    train_loader, test_loader = set_loader_competition(opt)

    prediction = []
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            model, criterion = set_model_competition_first(opt)
            print("loading checkpoint resume")
            checkpoint = torch.load(opt.resume, map_location=torch.device('cpu'))
            state_dict = clean_state_dict(checkpoint['model'])
            model.load_state_dict(state_dict, strict=True)
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            print(f"no checkpoint found at:{opt.resume}")
            return
        
        print("Test Recovery")
        out_list = test_multilabel(test_loader, model, criterion, opt)
        prediction = out_list
        df = pd.DataFrame(prediction, columns=['Path (Trial/Image Type/Subject/Visit/Eye/Image Name)',
                                            'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])

        # Lưu DataFrame thành tệp CSV
        df.to_csv('./submission/imagenet_olives_combined/prediction_swin_phase_1_24_epoch_phase_2_18_epoch.csv', index=False)
        print("Done!")
        return


    # training routine
    for i in range(0,1):
        model, criterion = set_model_competition_first(opt)

        # print(model)

        optimizer = set_optimizer(opt, model)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train_OCT_multilabel(train_loader, model, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}, loss:{:.2f}'.format(
                epoch, time2 - time1, acc, loss))
            
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_phase_2_epoch_{}_loss_{:.2f}.pth'.format(epoch, loss))
                save_model(model, optimizer, opt, epoch, save_file)
            
            out_list = test_multilabel(test_loader, model, criterion, opt)
            prediction = out_list

            df = pd.DataFrame(prediction, columns=['File_Name',
                                                    'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])

            # Lưu DataFrame thành tệp CSV
            save_csv = os.path.join(opt.save_folder, "prediction_{}_epoch_{}_loss_{}.csv".format(
                opt.backbone, epoch, loss
            ))
            df.to_csv(save_csv, index=False)


    #     out_list = test_multilabel(test_loader, model, criterion, opt)
    #     prediction = out_list

    # df = pd.DataFrame(prediction, columns=['Path (Trial/Image Type/Subject/Visit/Eye/Image Name)',
    #                                         'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])

    # # Lưu DataFrame thành tệp CSV
    # df.to_csv('./submission/prediction_swin_phase_1_18e_phase_2_3e.csv', index=False)


def clean_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
