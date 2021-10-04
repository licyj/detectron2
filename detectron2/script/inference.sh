#!/bin/bash
# Program:
#	Simplified inference mask process.
# History:
# 2021/08/31    Eric Chen	First release

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
TASK='0915ASL_men'
# FRAME_FOLDER='/home/Datasets/ASL/train/'${TASK}'/frame/'

CUSTOM_TEST_JSON_PATH='/home/Datasets/ASL/train/'${TASK}'/json/'
MODEL='runs/1004ASL_men60/model_final.pth'
# output_mask='/home/Datasets/mask/0916ASL-TSLpretrained'


echo `date`"===> strart inferencfe."
cd /home/detectron2/detectron2/
python inference.py $TASK $MODEL $CUSTOM_TEST_JSON_PATH 
echo `date`"===> inference done."