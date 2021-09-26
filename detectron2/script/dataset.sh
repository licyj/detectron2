#!/bin/bash
# Program:
#	Simplified inference mask process.
# History:
# 2021/08/31    Eric Chen	First release

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
TASK='0902segmentgood'
PICKUP_SRC='spreadthesign'
TRAIN_PATH='/home/Datasets/ASL/train/'${TASK}'/frame'
TRAIN_JSON_PATH='/home/Datasets/ASL/train/'${TASK}'/json'
VALID_VIDEO_PATH='/home/Datasets/ASL/train/'${TASK}'/'${TASK}'.json'
echo `date`"===> prepare for pickup dataset."
cd /home/detectron2/detectron2/tool/
python pickup2train.py $PICKUP_SRC  $VALID_VIDEO_PATH  $TRAIN_PATH  $TRAIN_JSON_PATH
echo `date`"===> prepare dataset done."