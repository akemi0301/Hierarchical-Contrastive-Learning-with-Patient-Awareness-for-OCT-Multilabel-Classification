from config.config_supcon_competition import parse_option
from utils.utils_supcon_competition import set_loader,set_model_contrast
from utils.utils import set_optimizer, adjust_learning_rate,save_model
import os
import time
# import tensorboard_logger as tb_logger
import torch 
from training_supcon.training_one_epoch_competition_patient_aware_bcva import train_Compeition_Patient_Aware_BCVA
import torch
def main():
    opt = parse_option()
    torch.cuda.empty_cache()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model_contrast(opt)

    # print(model)
    # sdv

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        loss = train_Compeition_Patient_Aware_BCVA(train_loader, model, criterion, optimizer, epoch, opt)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_model_{}_epoch_{}_loss_{:.2f}.pth'.format(opt.backbone, epoch, loss))
            save_model(model, optimizer, opt, epoch, save_file)

        torch.cuda.empty_cache()
        

    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()