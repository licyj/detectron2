#!/bin/bash

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

#=======================train args=======================
num_gpus=2
config_file='configs/0916ASL-TSLpretrained.yaml'
pretrained_model='runs/0916TSL-FBpretrained/model_final.pth'
train_set="('0915ASL_men',)"
val_set="('0901spreadthesign_men',)"
output_dir='runs/0917ASL-dataAug'
# output_dir='runs/test'
#=======================inference args=======================
train_img_folder='/home/Datasets/ASL/train/0915ASL_men/frame/'
jsons='/home/Datasets/ASL/train/0915ASL_men/json'
inference_model='runs/0917ASL-dataAug/model_final.pth'
mask_path='/home/Datasets/mask/0917ASL-dataAug/'
output_video='/home/Datasets/result/0917ASL-dataAug/'

cd /home/detectron2/detectron2
sh script/train.sh ${num_gpus} ${config_file} ${pretrained_model} ${train_set} ${val_set} ${output_dir}

# # infernce
sh script/inference.sh $inference_model $jsons
# # output video
python tool/apply_mask.py -f $train_img_folder \
                          -m $mask_path \
                          -t $output_video