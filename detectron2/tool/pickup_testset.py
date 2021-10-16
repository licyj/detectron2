from genericpath import exists
import os
import pathlib
import json
import argparse
import random
import shutil
'''
    WLASL_json: <list> of glosses, each gloss would be <dict>
        gloss 1
            key1: gloss <str>
            key2: instances <list>
                for each instance, there are some attributes
                    bbox
                    fps
                    frame_end
                    frmae_start
                    instance_id
                    signer_id
                    source
                    split
                    url
                    variation_id
                    video_id
    History:
        Eric Chen   2021/10/11  'first release'
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_label', type=str, help='path/to/train/label.txt')
    parser.add_argument('testset_ize', type=int, help='number of testset-size')
    # parser.add_argument('src_frame', type=str, help='path/to/put/test_set/frames')
    args = parser.parse_args()
    return args


def get_label_pair(file):
    '''
        input:
            label.txt path
        output:
            label_dict: <dict> video_id  gloss
    '''
    label_dict = {}
    f = open(file, 'r')
    txt_lines = f.readlines()
    # video_id \t gloss
    for line in txt_lines:
        line = str(line).replace('\n','')
        video_id, label = line.split('\t')
        label_dict[str(video_id)] = str(label)
    return label_dict 


def video_exist(v_id):
    '''
        function:
            check video exist in WLASL or not.
    '''
    return True if pathlib.Path(os.path.join('../../Datasets/WLASL/videos/',v_id+'.mp4')).exists() else False


def get_testset_dict(f, train_glosses):
    '''
        function:
            find available test set using exist gloss 
        outout:
            pickup_testSet: <dict> 
                gloss:<str>, available_videos:<list> 
    '''
    pickup_dict = {}
    train_source =['spreadthesign']
    for entry in f:
        gloss = entry['gloss']
        if gloss in train_glosses:
            pickup_dict[gloss] = list()
            for inst in entry['instances']:
                # id, source 
                v_id = inst['video_id']
                v_source = inst['source']
                if v_source not in train_source and video_exist(v_id):
                    pickup_dict[gloss].append(v_id)
        else:
            continue
    return pickup_dict


def dump_label_txt(dict, dst_path):
    label_path = os.path.join(dst_path, 'label.txt')
    with open(label_path, 'w') as f:
        for k, v in sorted(dict.items()):
            f.write(str(k)+'\t'+str(v)+'\n')


def main():
    '''
        function:
            - get test set label.txt
            - copy test set frames to dst path
        input:
            - training label.txt to get training gloss
            - 
    '''
    args = get_args()
    WLASL_json = '/home/Datasets/WLASL_v0.3.json'
    SRC_WLASL = '../../Datasets/WLASL/frame/'
    DST_PATH = '../../Datasets/ASL/To_111/testset/'
    with open(WLASL_json, 'r') as _:
        json_file = json.load(_)
    # video_id, gloss 
    label_dict = get_label_pair(file= args.train_label)
    pickup_dict = get_testset_dict(json_file, label_dict.values())
    
    #TODO: choose available video and copy to dst
    label_ = {} 
    for __ in range(args.testset_ize):
        rand_gloss = random.choice(list(pickup_dict))
        rand_v_id = random.choice(pickup_dict[str(rand_gloss)])
        dst = os.path.join(DST_PATH, 'frames/{}'.format(str(rand_v_id)))
        src = os.path.join(SRC_WLASL, rand_v_id)
        print(rand_gloss,'\n', rand_v_id,'\n' , dst)
        shutil.copytree(src, dst)
        label_[str(rand_v_id)] = rand_gloss
    dump_label_txt(label_, DST_PATH)

if __name__ == '__main__':
    main()