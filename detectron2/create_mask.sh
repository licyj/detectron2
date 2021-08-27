#!/bin/bash
# Program:
#       This program including split video, create testset json file and inference testset mask.
# Histort:
#       2021/08/27 Eric Chen First release

########### PATH ###########
src_video_path=
tar_frame_dir=
TEST_IMG_FOLDER=
output_mask=
testset_json_path=
TARGET_JSON_FOLDER=
############################

PYTHON=`which python3`
cd /home/detectron2/detectron2/tool
echo `ls`

# {source_video_path} {tgt_frame_dir}  
python3 video2frame.py 
# {base_frame_dir} {tgt_json_base_dir}
python3 testset2json.py

cd /home/detectron2/detectron2
# {} {}
python3 inference.py 