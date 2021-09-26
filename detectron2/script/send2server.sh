#!/bin/bash
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

'''
    function:
        Send frames, indice.txt, label.txt to server
    usage:
        1. first execute get_ASL_label.py, get_ASL_indice.py
        2. modify server_path data_path parameter 
'''
#============ parameter #============
frames='/home/Datasets/ASL/train/'
mask='/home/Datasets/mask/'
server='140.115.54.111'
#============#============#============

cd /home/Datasets/ASL/To_111/
sudo rsync -e 'ssh -p 9487' $frmaes $mask ./indice.txt ./label.txt \
    msp@$server:/mnt/2TDisk1/data/wjm/Datasets/ASL/