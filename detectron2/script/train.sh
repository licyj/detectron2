#!/bin/bash
# Program:
#	Simplified training process.
# History:
    # 2021/08/31    Eric Chen	First release
    # 2021/09/17    Eric Chen   change argument sciprt version
# Reference:
#   https://lyt0112.pixnet.net/blog/post/308636887-shell-script-%E5%AD%97%E4%B8%B2%E5%90%88%E4%BD%B5
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

'''
    function:
        train detectron2
    args:
        $1 num-gpus
        $2 path/to/config-file
        $3 path/to/output_dir
'''

echo `date`"===> training start"
cd /home/detectron2/detectron2/
python train_net.py --num-gpus $1\
                    --config-file $2\
                    MODEL.WEIGHTS $3\
                    DATASETS.TRAIN $4\
                    DATASETS.TEST $5\
                    OUTPUT_DIR $6
echo `date`"===> training done."
