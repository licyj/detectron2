import os
import json
import argparse
from sys import version
import pandas as pd
from get_ASL_label import get_trans_dict, get_id_label_pair
from get_ASL_indice import get_indice
'''
    function:
        To get label.txt, indice.txt, frame
    
    label.json
'''


def get_dataset_info():
    dataset_info = pd.read_csv(open('/home/Datasets/ASL_information.csv', 'r'))
    dataset_info = dataset_info[['video_id','gloss']]
    return dataset_info


def get_args():
    parser = argparse.ArgumentParser()
    #https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument('--labeled_json',type=str, nargs='+', help='root/of/json_need_merge')
    # parser.add_argument('--label_txt',type=str, nargs='+', help='path/to/label_need_merge')
    parser.add_argument('--ori_transformFile',type=str, nargs='+', help='path/to/label_need_merge')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dataset_info = get_dataset_info()
    trans_dict = {}
    #get get root of label_json
    for transform_txt in args.ori_transformFile:
        temp={}
        with open(transform_txt,'r') as f:
            temp.update(get_trans_dict(f))
        frame_path = os.path.join(str(args.ori_transformFile).split('/'), 'frame')
        print(frame_path)
        assert 0
        for k, v in list(trans_dict.items()):
            if v not in sorted(os.listdir(frame_path)):
                print('key {} is not in label dir.'.format(v))

    # different labeled source 
    for source in sorted(os.listdir(args.label_json)):
        # delete video key pair that is not selected
        for k,v in list(trans_dict.items()):
            if v not in sorted(os.listdir(source)):
                print('key {} is not in label dir.'.format(v))
                # del trans_dict[k]
        print(source)
        get_id_label_pair(dataset_info, transform_table=trans_dict)
        # for video in sorted(os.listdir(source)):


if __name__ == '__main__':
    main()