#  python classification/train_OLIVES.py --data ./data_OLIVES/ \
#  --model 'resnet50' \
#  --train-listfile "train_listfile_main.json" \
#  --val-listfile val_listfile.json \
#  --class-map-file "class_map.json" \
#  --num-classes 17 \
#  --batch-size 100\
#  --repeating-product-file "repeating_product_ids.csv"\
#  --learning_rate 0.5 --temp 0.1 \
#  --dist-url 'tcp://localhost:10001' \
#  --multiprocessing-distributed \
#  --world-size 1 --rank 0 --cosine

# python classification/train_deepfashion.py --data ./data_OLIVES/ \
#  --model 'tresnetm_v2' \
#  --train-listfile "train_listfile_main.json" \
#  --val-listfile val_listfile.json \
#  --class-map-file "class_map.json" \
#  --num-classes 17 \
#  --batch-size 48\
#  --input-size 448\
#  --repeating-product-file "repeating_product_ids.csv"\
#  --learning_rate 0.5 --temp 0.1 \
#  --dist-url 'tcp://localhost:10001' \
#  --multiprocessing-distributed \
#  --world-size 1 --rank 0 --cosine

# python training_hierarchical_supcon/train_OLIVES_phase_1.py --data ./hierarchical_phase1/data_OLIVES/ \
#  --backbone 'tresnetl_v2' \
#  --train-listfile "train_listfile_OLIVES_final_csv_1.json" \
#  --class-map-file "class_map.json" \
#  --num-classes 17 \
#  --batch-size 40\
#  --img_size 448\
#  --print_freq 10\
#  --learning_rate 0.1 --temp 0.1 \
#  --dist-url 'tcp://localhost:10001' \
#  --multiprocessing-distributed \
#  --world-size 1 --rank 0 --cosine

python training_hierarchical_supcon/train_OLIVES_phase_1.py --data ./hierarchical_phase1/data_OLIVES/ \
 --backbone 'swin_T_224_22k' \
 --train-listfile "train_listfile_OLIVES_final_csv_1.json" \
 --class-map-file "class_map.json" \
 --num-classes 17 \
 --batch-size 256\
 --img_size 224\
 --print_freq 10\
 --learning_rate 0.1 --temp 0.1 \
 --dist-url 'tcp://localhost:10001' \
 --multiprocessing-distributed \
 --world-size 1 --rank 0 --cosine