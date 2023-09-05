# backbone Swin transformer
# python training_main/clinical_sup_contrast_competition.py \
#     --patient_split 1 --img_size 384 --hidden_dim 2048 \
#     --keep_input_proj --dim_feedforward 8192 --num_methods 2 --method1 'bcva' --method2 'cst' \
#     --dataset 'Prime_TREX_DME_Fixed' --epochs 30 --device 'cuda:0' \
#     --pretrained --grad_visualize  --learning_rate 0.0005\
#     --train_image_path '../data/Datasets_swinIR' \
#     --backbone 'swin_B_384_22k' --batch_size 8 --save_freq 2

# backbone Tresnet v2
python training_main/clinical_sup_contrast.py \
    --patient_split 1 --img_size 448 --hidden_dim 2048 \
    --keep_input_proj --dim_feedforward 8192 --num_methods 2 --method1 'bcva' --method2 'cst' \
    --dataset 'Prime_TREX_DME_Fixed' --epochs 30 --device 'cuda:0' \
    --pretrained --grad_visualize\
    --train_image_path '../data/Datasets' \
    --backbone 'tresnetl_v2' --batch_size 28 --save_freq 2