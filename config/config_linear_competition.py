
import argparse
import math
import os
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--save_folder', type=str, default='./save_phase_2')

    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--patient_lambda', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=int, default=1, help='data parallel')
    
    # model dataset
    parser.add_argument('--train_csv_path', type=str, default='train data csv')
    parser.add_argument('--val_csv_path', type=str, default='val data csv')
    parser.add_argument('--test_csv_path', type=str, default='test data csv')

    parser.add_argument('--train_image_path', type=str, default='/data/Datasets')
    parser.add_argument('--val_image_path', type=str, default='/data/Datasets')
    parser.add_argument('--test_image_path', type=str, default='/data/Datasets')

    parser.add_argument('--img_dir', type=str, default='image directory')
    parser.add_argument('--model_type', type=str, default='bcva')
    parser.add_argument('--multi', type=int, default=0)
    parser.add_argument('--noise_analysis', type=int, default=0)
    parser.add_argument('--severity_analysis', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='Competition')

   


# * Transformer
    parser.add_argument('--competition', type=int, default=1)

    parser.add_argument('--img_size', default=384, type=int,
                        help='size of input images')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model for backbone. default is False. ') 
    parser.add_argument('--num_class', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--with_transformer_head', action='store_true',
                        help='use transformer head attached with backbone swin or cvt. Do not use when training constrastive learning') 


    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
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

    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--ford_region',type = int,default = 0,
                        help='Training on 6 region classes or not')
    parser.add_argument('--percentage', type=int, default=100,
                        help='Percentage of Biomarker Training Data Utilized')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--backbone_training', type=str, default='BCVA',
                        help='manner in which backbone was trained')
  
    # method visualize explaination AI
    parser.add_argument('--grad_visualize', dest='grad_visualize', action='store_true',
                        help='use gradcam to visualize for model. default is False. ') 
    parser.add_argument('--grad_method', type=str, default='scorecam')
    parser.add_argument('--tsne_visualize', type=bool, default=True)
    

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.img_visualize = opt.save_folder+'img_visualize/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.backbone+'-q2l', opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)  


    return opt