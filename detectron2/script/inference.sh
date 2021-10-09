#!/bin/bash
# Program:
#	Simplified inference mask process.
# History:
# 2021/08/31    Eric Chen	First release

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
model='runs/ASL_men_2class/model_0001999.pth'
root_frame='/home/Datasets/ASL/train/ASL_men189/frame'
root_mask='/home/Datasets/mask/ASL_men_2_class'
train_json='/home/Datasets/ASL/train/ASL_men189/train_ASL_men189.json'
# CUSTOM_TEST_JSON_PATH='/home/Datasets/ASL/train/ASL_men189/json/'

echo `date`"===> strart inferencfe."
cd /home/detectron2/detectron2/
python inference.py $model $root_frame $root_mask  $train_json
echo `date`"===> inference done."