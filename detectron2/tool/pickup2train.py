import os, sys
import json
import pandas as pd
from WLASL2csv import get_wlasl
import shutil
import argparse
'''
    Program:
            This porgram is for looking up source of best prediction example 
            Analysis video 
    Histroy:
            2021/08/28 Eric Chen First release
    Reference:
            import self module
                https://www.itread01.com/content/1545229284.html
'''
WLASL_INFO_PATH = '/home/Datasets/WLASL_v0.3.json'
VIDEO_ALL_PATH = '/home/Datasets/WLASL/frame/'
JSON_ALL_PATH =  '/home/Datasets/WLASL/json/'
### need to change ### 
# VALID_VIDEO_PATH =''
# TRAIN_PATH      = '/home/Datasets/ASL/train/0901valid_video'
# TRAIN_JSON_PATH = '/home/Datasets/ASL/train/0901valid_video/json'
'''
    # source: spreadthesign
    best_sample = ['00853', '01460', '04040', '04849', '05629', '58589']
    # source: signingsavvy
    good_sample = ['01962', '07245', '04600', '09155', '09468', '09924', '24954']
    # source: spreadthesign
    not_bad = ['02711', '11876', '11994', '16925']
'''

'''
    source: spreadthesign, total: 1583 
    signer 0
        total: 189
    signer 2 
        total: 561
    video: 00853 source spreadthesign signer 0
    video: 05629 source spreadthesign signer 2
    video: 01460 source spreadthesign signer 2
    video: 04040 source spreadthesign signer 2
    video: 04849 source spreadthesign signer 2
    video: 07245 source signingsavvy signer 11
    video: 01962 source signingsavvy signer 11
    video: 09155 source signingsavvy signer 11
    video: 04600 source signingsavvy signer 11
    video: 09468 source signingsavvy signer 11
    video: 09924 source signingsavvy signer 11
    video: 24954 source signingsavvy signer 10
'''
def cal_valid_video(args, dataset):
    # PICKUP_SRC = ['spreadthesign', 'signingsavvy']
    PICKUP_SRC = ['spreadthesign']
    spreadthesign_signer = [0]
    # signingsavvy_singer = [10, 11]
    valid = []
    for source in PICKUP_SRC:
        PICKUP_DICT = {}
        PICKUP_DICT['source'] = source
        PICKUP_DICT['videos'] = []
        for i in dataset:
            # signers =  spreadthesign_signer if source == 'spreadthesign' \
            #                                 else signingsavvy_singer
            signers =  spreadthesign_signer
            if i['video_id'] in os.listdir(VIDEO_ALL_PATH)\
                                and i['source'] == source\
                                and i['signer_id'] in signers:
                PICKUP_DICT['videos'].append(i['video_id'])
                signer_idx = 'signer_'+ str(i['signer_id'])
                if signer_idx not in PICKUP_DICT.keys():
                    PICKUP_DICT[signer_idx] = 0 
                    PICKUP_DICT[signer_idx] += 1
                else:
                    PICKUP_DICT[signer_idx] += 1
        valid.append(PICKUP_DICT)
    with open(args.VALID_VIDEO_PATH, 'w') as f:
        json.dump(valid, f)
    return valid

def prepare_trainset(args, validVideo):
    '''
        input: 
            dictionary of recording valid sources and videos 
        introduction:
            this function's purpose is to copy valid video frames to trainset
    '''
    for src in validVideo:
        source = src['source']
        videos = src['videos']
        # videos = ['01418', '04040', '04155', '04168', '04580', '04796', \
        #             '05556', '05629', '05705', '05845', '05853', '06649', '06787', \
        #             '09179', '09487', '09950']
        for video in videos:
            WLASL_frame = os.path.join(VIDEO_ALL_PATH, video)
            WLASL_JSON = os.path.join(JSON_ALL_PATH, video)
            train_frame = os.path.join(args.TRAIN_PATH, source, video)
            train_json = os.path.join(args.TRAIN_JSON_PATH, source, video)
            shutil.copytree(WLASL_frame, train_frame)
            shutil.copytree(WLASL_JSON, train_json)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("PICKUP_SRC",
                        type=str,
                        help="task name")
    parser.add_argument("VALID_VIDEO_PATH",
                        type=str,
                        help="task name")
    # parser.add_argument("signer",
    #                     type=str,
    #                     help="task name")
    parser.add_argument("TRAIN_PATH",
                        type=str,
                        help="json files within testset")
    parser.add_argument("TRAIN_JSON_PATH",
                        type=str,
                        help="model name")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if not os.path.exists(args.TRAIN_PATH):
        os.makedirs(args.TRAIN_PATH)
    if not os.path.exists(args.TRAIN_JSON_PATH):
        os.makedirs(args.TRAIN_JSON_PATH)
    valid_dict = cal_valid_video(args, dataset= get_wlasl(json_path= WLASL_INFO_PATH))
    print('calculate valid dictionary done.')
    with open(args.VALID_VIDEO_PATH,'r') as f:
        validVideo = json.load(f)
    prepare_trainset(args, validVideo)

if __name__ == '__main__':
    main()