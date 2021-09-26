#!/bin/bash
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

#=================== paths ======================
'''
    function: 
        convert png to train.json
    args:
        $1: path/to/mask_folder
        #2: path/to/train.json
'''
echo `date` "preparing train.json"
cd /home/detectron2/detectron2/
python tool/png2mrcnn_label.py $1 $2
echo `data` "generating train.json done."