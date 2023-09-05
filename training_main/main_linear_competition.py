from __future__ import print_function
from config.config_linear_competition import parse_option
# from training_linear.training_one_epoch_ckpt import main
# from training_linear.training_one_epoch_fusion import main_supervised_fusion
# from training_linear.training_one_epoch_supervised import main_supervised
# from training_linear.training_one_epoch_supervised_multilabel import main_supervised_multilabel
# from training_linear.training_one_epoch_fusion_multilabel import main_supervised_multilabel_fusion
# from training_linear.training_one_epoch_ckpt_multi import main_multilabel
# from training_linear.training_one_epoch_ckpt_bce import main_bce
# from training_linear.training_one_epoch_transformer import main_transformer
# from training_linear.training_one_epoch_transformer_multilabel import main_transformer_multilabel
# from training_linear.training_one_epoch_ckpt_student_teacher import main_student_teacher

from training_linear.training_one_epoch_ckpt_competition import main_multilabel_competition
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

if __name__ == '__main__':
    opt = parse_option()

    if (opt.competition == 1):
        main_multilabel_competition()

    