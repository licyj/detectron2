from argparse import ArgumentParser
import os
from pathlib import Path
import argparse
import json

def get_indice(args):
    '''
        funciton:
            return indice: <dict>
        input:
            root/of/label_json <Path>
        output:
            indice dictionary: current_video, start_frame, end_frame
    '''
    indice_dict={}
    for video in sorted(os.listdir(args.label_json)):
        indice_dict[str(video)]=[]
        start_frame = 0
        end_frame = 0
        started = False
        ended = False
        for file in sorted(os.listdir(os.path.join(args.label_json, video))):
            with open(os.path.join(args.label_json, video, file), 'r') as __:
                f = json.load(__)
            if len(f['labels'])>1 and not started:
                started = True
                start_frame = int(file.split('__')[0])
                indice_dict[video].append(start_frame)
            if len(f['labels'])==1 and started and not ended:
                ended = True
                end_frame = int(file.split('__')[0])
                indice_dict[video].append(end_frame)
                break
    return indice_dict

def dict2indice_txt(args,indice):
    '''
        function:
            write out indice.txt
    '''
    if not os.path.exists('/home/Datasets/ASL/To_111/{}'.format(str(args.label_json).split('/')[-2])):
        os.mkdir('/home/Datasets/ASL/To_111/{}'.format(str(args.label_json).split('/')[-2]))
    outPath = os.path.join('/home/Datasets/ASL/To_111/{}'.format(str(args.label_json).split('/')[-2]),'label.txt')
    with open(outPath, 'w') as f:
        for video in sorted(indice.keys()):
            f.write(video +'\t'+ str(indice[video][0]) +'\t'+ str(indice[video][1]) + '\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_json', type=str, help='path/to/label_json_path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    indice = get_indice(args)
    dict2indice_txt(args, indice)


if __name__ == '__main__':          
    main()