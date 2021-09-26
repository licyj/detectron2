from argparse import ArgumentParser
import os
from pathlib import Path
import argparse
import pandas as pd
from pandas.core.accessor import DirNamesMixin
'''
    function:
        output label file: video_id  video_label
'''

def get_trans_dict(transfromFile):
    '''
        function:
            get dictionary of transformed information
        input:
            transformation file
        output:
            trans_dict: <dict>
    '''
    trans = transfromFile.readlines()
    trans_dict={}
    for line in trans:
        line = str(line).replace('\n','')
        ori_id, video_id = line.split('\t')
        trans_dict[str(ori_id)] = str(video_id)
    return trans_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('transformFile',type=str, help='path/to/transform_txt')
    args = parser.parse_args()
    return args


def get_id_label_pair(dataset_info, transform_table):
    '''
        input:
            dataset_info:    <data_frame>   ASL information in csv
            transform_table: <dict>         ori_id, video_id
    '''
    labels = {}
    for ori_id in sorted(transform_table.keys()):
        current_id = transform_table[ori_id]
        matched_row = (dataset_info.loc[dataset_info['video_id'].isin([ori_id])])
        gloss = matched_row['gloss'].values[0]
        labels[current_id] = str(gloss)
    return labels


def dump2txt(outPath, id_label_pair):
    f = open(outPath, 'w') 
    for video_id in id_label_pair.keys():
        f.write(video_id+'\t'+id_label_pair[video_id]+'\n')
    f.close


def main():
    args = get_args()
    outPath = os.path.join('/home/Datasets/ASL/To_111/', str(args.transformFile).split("/")[-2], 'test_label.txt')
    dataset_info = pd.read_csv(open('/home/Datasets/ASL_information.csv', 'r'))
    dataset_info = dataset_info[['video_id','gloss']]
    # transformation info
    trans_dict = get_trans_dict(transfromFile= open(args.transformFile, 'r'))
    id_label_pair = get_id_label_pair(dataset_info, transform_table=trans_dict)
    dump2txt(outPath, id_label_pair)

if __name__ == '__main__':
    main()