# backbone Swin transformer
#  --hidden_dim 2048  --keep_input_proj --dim_feedforward 8192 --num_methods 2 --method1 'bcva' --method2 'cst' 
# python training_main/clinical_sup_constrast_competition.py \
#     --img_size 384 \
#     --epochs 5 --device 'cuda:0' \
#     --print_freq 10\
#     --pretrained --grad_visualize  --learning_rate 0.001\
#     --train_image_path '../data/Datasets' \
#     --backbone 'swin_L_384_22k' --batch_size 8 --save_freq 1

# -------------- MAX BATCH SIZE (LOSS NOT NAN) ------------------
#--pretrained 
python training_main/clinical_sup_constrast_competition.py \
    --img_size 384 \
    --epochs 30 --device 'cuda:0' \
    --print_freq 10\
    --pretrained \
    --learning_rate 0.001\
    --train_image_path '../data/Datasets' \
    --backbone 'tresnetl_v2' --batch_size 32 --save_freq 2

# python training_main/clinical_sup_constrast_competition.py \
#     --img_size 448 \
#     --epochs 5 --device 'cuda:0' \
#     --print_freq 10\
#     --pretrained --learning_rate 0.001\
#     --train_image_path '../data/Datasets' \
#     --backbone 'tresnetm_v2' --batch_size 56 --save_freq 2

# python training_main/clinical_sup_constrast_competition.py \
#     --img_size 224 \
#     --epochs 5 --device 'cuda:0' \
#     --print_freq 10\
#     --pretrained --learning_rate 0.001\
#     --train_image_path '../data/Datasets' \
#     --backbone 'swin_T_224_22k' --batch_size 256 --save_freq 2
