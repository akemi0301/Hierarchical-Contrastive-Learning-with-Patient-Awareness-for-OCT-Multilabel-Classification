import argparse
import math
import os

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--training_phase', type=str, default='training_phase_1',
                        choices=['training_phase_1'], help='specify phase \
                        for training (check models/query2label_models/backbone.py)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--patient_lambda', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--cluster_lambda', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_csv_path', type=str, default='train data csv')
    parser.add_argument('--test_csv_path', type=str, default='test data csv')
    parser.add_argument('--train_image_path', type=str, default='train data csv')
    parser.add_argument('--test_image_path', type=str, default='test data csv')
    parser.add_argument('--results_dir_contrastive', type=str, default='/home/kiran/Desktop/Dev/SupCon_OCT_Clinical/results.txt')
    parser.add_argument('--percentage', type=int, default=10,
                        help='momentum')
    parser.add_argument('--discrete_level', type=int, default=10,
                        help='discretization Level')
    parser.add_argument('--parallel', type=int, default=1, help='data parallel')
    # model dataset
    
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=128, help='parameter for RandomResizedCrop')

    # advanced setting for model
    # * Transformer
    
    parser.add_argument('--img_size', default=384, type=int,
                        help='size of input images')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model for backbone. default is False. ') 
    parser.add_argument('--num_class', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    # parser.add_argument('--with_transformer_head', action='store_true',
    #                     help='use transformer head attached with backbone swin or cvt. Do not use when training constrastive learning') 


    # parser.add_argument('--enc_layers', default=1, type=int, 
    #                     help="Number of encoding layers in the transformer")
    # parser.add_argument('--dec_layers', default=2, type=int,
    #                     help="Number of decoding layers in the transformer")
    # parser.add_argument('--dim_feedforward', default=8192, type=int,
    #                     help="Intermediate size of the feedforward layers in the transformer blocks")
    # parser.add_argument('--hidden_dim', default=2048, type=int,
    #                     help="Size of the embeddings (dimension of the transformer)")
    # parser.add_argument('--dropout', default=0.1, type=float,
    #                     help="Dropout applied in the transformer")
    # parser.add_argument('--nheads', default=4, type=int,
    #                     help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")


    # method
    # parser.add_argument('--num_methods', type=int, default=0,
    #                     help='choose method')
    # parser.add_argument('--method1', type=str, default='n',
    #                     help='choose method')
    # parser.add_argument('--method2', type=str, default='n',
    #                     help='choose method')
    # parser.add_argument('--method3', type=str, default='n',
    #                     help='choose method')
    # parser.add_argument('--method4', type=str, default='n',
    #                     help='choose method')
    # parser.add_argument('--method5', type=str, default='n',
    #                     help='choose method')
    # parser.add_argument('--alpha', type=float, default=1,
    #                     help='choose method')
    parser.add_argument('--gradcon_labels', type=str, default='',
                        help='choose method')
    # parser.add_argument('--patient_split', type=int, default=1,
    #                     help='choose method')
    parser.add_argument('--quantized', type=int, default=0,
                        help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # method visualize explaination AI
    parser.add_argument('--grad_visualize', dest='grad_visualize', action='store_true',
                        help='use gradcam to visualize for model. default is False. ') 
    parser.add_argument('--grad_method', type=str, default='scorecam')
    parser.add_argument('--tsne_visualize', dest='tsne_visualize', action='store_true',
                        help='use gradcam to visualize for model. default is False. ') 
    
    #save weight
    parser.add_argument('--save_model', type=str, default=None)

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    # if opt.dataset == 'path':
    #     assert opt.data_folder is not None \
    #            and opt.mean is not None \
    #            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save_phase_1/SupCon/{}_models'.format(opt.backbone)
    opt.tb_path = './save_phase_1/SupCon/{}_tensorboard'.format(opt.backbone)
    opt.img_visualize = './img_visualize/bcva_method'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'bcva_patient_aware_backbone_{}_lr_{}_decay_{}_bsz_{}_trial{}'.\
        format(opt.backbone, opt.learning_rate, opt.weight_decay,
                opt.batch_size, opt.trial)
    # opt.model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_lr_{}_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_{}'. \
    #     format(opt.method1, opt.method2, opt.method3, opt.method4, opt.method5, opt.alpha, opt.patient_split,opt.discrete_level,
    #            opt.dataset, opt.backbone, opt.learning_rate,
    #            opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.gradcon_labels, opt.quantized)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt