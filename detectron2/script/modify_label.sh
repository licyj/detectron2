#!/bin/bash
# Program:
#	Simplified create modify label json files process.
# History:
# 2021/09/01    Eric Chen	First release

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
TASK='0901spreadthesign_women'
frame_path='/home/Datasets/ASL/train/'${TASK}'/frame/long_sleeve/*'
mask_path='/home/Datasets/mask/'${TASK}
target_json_path="/home/Datasets/label_json/0909_"${TASK}

# echo "====> start create label json"
# cd /home/detectron2/detectron2/tool
# python png2json.py $mask_path $target_json_path
# echo "====> create label json done."

echo "====> zip json files start."
mkdir '/home/Datasets/zip/'$TASK
cd $target_json_path
mkdir '/home/Datasets/zip/'$TASK'/json'
mkdir '/home/Datasets/zip/'$TASK'/img'
cp -r ./* '/home/Datasets/zip/'$TASK'/json'
cp -r $frame_path '/home/Datasets/zip/'$TASK'/img'
cd '/home/Datasets/zip/'
zip -r $TASK'.zip' $TASK/*
echo "====> zip json files done."
