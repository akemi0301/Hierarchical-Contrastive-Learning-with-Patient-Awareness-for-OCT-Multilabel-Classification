#!/bin/bash
#  --ckpt '/mnt/HDD/Chau_Truong/SupCon_OCT_Clinical/save/SupCon/Prime_TREX_DME_Fixed_models/bcva_cst_n_n_n_1_1_10_Prime_TREX_DME_Fixed_lr_None_0.05_decay_0.0001_bsz_8_temp_0.07_trial_0__0/ckpt_epoch_18.pth'\

python training_main/main_linear_competition.py\
 --dataset 'Competition' \
 --ckpt '/mnt/HDD/Chau_Truong/SupCon_OCT_Clinical/pretrained_hierarchical_constrastive/tresnetl_v2/checkpoint_loss_2.2127540590925525_epoch_0038_.pth.tar' \
 --competition 1 --backbone "tresnetl_v2" --with_transformer_head \
 --save_folder './save_phase_2/hierarchical_constrastive_pretrained/tresnetl_v2_epoch_38/bsz_80/' \
 --num_class 6 --epochs 30 --save_freq 2 --print_freq 50 --batch_size 80 \
 --img_size 448 --hidden_dim 2048 \
 --learning_rate 0.001 \
 --keep_input_proj --dim_feedforward 8192 \
 --amp \
 --train_csv_path '/mnt/HDD/Chau_Truong/data/Datasets/Training_Biomarker_Data_merge_updated.csv' \
 --val_csv_path '' \
 --test_csv_path '/mnt/HDD/Chau_Truong/data/Datasets/Phase2_submission_template.csv' \
 --train_image_path '/mnt/HDD/Chau_Truong/data/Datasets' \
 --val_image_path '' \
 --test_image_path '/mnt/HDD/Chau_Truong/data/Datasets/TEST_PHASE2/VIPCUPData_Phase2/DME'