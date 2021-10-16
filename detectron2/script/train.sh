#!/bin/bash
# Program:
#	Simplified training process.
# History:
    # 2021/08/31    Eric Chen	First release
    # 2021/09/17    Eric Chen   change argument sciprt version
# Reference:
#   https://lyt0112.pixnet.net/blog/post/308636887-shell-script-%E5%AD%97%E4%B8%B2%E5%90%88%E4%BD%B5

'''
    function:
        train detectron2
    args:
        $1 num-gpus
        $2 path/to/output_dir
'''

echo `date`"===> training start"
cd /home/detectron2/detectron2/
python train_net.py --num-gpus $1 --config-file $2 OUTPUT_DIR $3
                    # MODEL.WEIGHTS $3\
                    # DATASETS.TRAIN $4\
                    # DATASETS.TEST $5\
echo `date`"===> training done."
