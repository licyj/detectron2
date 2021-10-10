import os
import pathlib

label_root = '../../Datasets/label_json/ASL_women/'
frame_root = '../../Datasets/ASL/train/ASL_women/frame/long_sleeve/'

for _l in os.listdir(label_root):
    for __f in os.listdir(frame_root):
        if len(_l) is not len(__f):
            print(_l)