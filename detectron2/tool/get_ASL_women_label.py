from argparse import ArgumentParser
import os
from pathlib import Path
import argparse
import pandas as pd


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


def get_id_label_pair(dataset_info, transform_table):
    '''
        input:
            dataset_info:    <data_frame>   ASL information in csv
            transform_table: <dict>         ori_id, video_id
        output:
            labels:          <dict>         current_id, gloss 
    '''
    labels = {}
    for ori_id in sorted(transform_table.keys()):
        current_id = transform_table[ori_id]
        matched_row = (dataset_info.loc[dataset_info['video_id'].isin([ori_id])])
        gloss = matched_row['gloss'].values[0]
        labels[current_id] = str(gloss)
    return labels


def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def get_women_transformDict(trans_dict, rename_dict):
    '''
        input:
            trans_dict
                dict: <ori_id, temp_id>
            rename_dict
                dict: <temp_id, final_id>
        output:
            out_dict
                dict: <ori_id, current_id>
    '''
    out_dict = {}
    for key, val in sorted(rename_dict.items()):
        ori_id = get_key(key, trans_dict)
        out_dict[str(ori_id)] = str(val)
    return out_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('videoTransform',type=str, help='path/to/videoTransform_txt')
    parser.add_argument('rename_txt',type=str, help='path/to/rename_txt')
    args = parser.parse_args()
    return args


def dump2txt(outPath, id_label_pair):
    f = open(outPath, 'w')
    for video_id in id_label_pair.keys():
        f.write(video_id+'\t'+id_label_pair[video_id]+'\n')
    f.close()


def main():
    args = get_args()
    outPath = os.path.join('/home/Datasets/ASL/To_111/', str(args.videoTransform).split("/")[-2], 'women_label.txt')
    dataset_info = pd.read_csv(open('/home/Datasets/ASL_information.csv', 'r'))
    dataset_info = dataset_info[['video_id','gloss']]
    # transformation info
    trans_dict = get_trans_dict(transfromFile= open(args.videoTransform, 'r'))
    rename_dict = get_trans_dict(transfromFile= open(args.rename_txt, 'r'))
    _women_trans_dict = get_women_transformDict(trans_dict, rename_dict)

    id_label_pair = get_id_label_pair(dataset_info, transform_table = _women_trans_dict)
    dump2txt(outPath, id_label_pair)


if __name__ == '__main__':
    main()