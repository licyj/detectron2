from argparse import ArgumentParser
import argparse
import json
import os
import sys

from yaml.events import NodeEvent
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent import futures
from signal import *
import time
'''
    function:
        Convert labelled or revised label json file to mask.png
    History:
        2021/09/08 Eric "deal mask index is exceed than 720." =>  "remove 'data_type' from main()"
'''


def thread_run(data):
    frames_folder, label = data
    with open(os.path.join(frames_folder, label)) as f:
        json_data = json.load(f)
    return json_data

def process_run(data):
    json_data, dst_folder = data
    label_order = ['head', 'right_hand', 'left_hand', 'others']
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask_rows, mask_cols = mask.shape
    # print(json_data)
    # print(json_data)
    # assert 0
    if 'labels' not in json_data.keys():
        print(json_data)
    
    for labels in json_data['labels']:
        filename = json_data['image_filename']+'.png'
        if labels is not None:
            # print(labels.keys())
            for coord in labels['regions'][0]:
                try:
                    if coord['x']>=int(mask_cols):
                        coord['x'] = int(mask_cols) -1
                    if coord['y']>=int(mask_rows):
                        coord['y'] = int(mask_rows) -1
                    # 0:bg, 1:head, 2:right_hand, 3:left_hand
                    mask[int(coord['y']), int(coord['x'])] = label_order.index(labels['label_class']) + 1
                except Exception as e:
                    print('wrong file:',filename)
                    # print(os.path.join(dst_folder, filename))
                    # print(coord['y'])
                    # print(coord['x'])
                    raise e
        cv2.imwrite(os.path.join(dst_folder, filename), mask)

def process(frames_folder, dst_folder):
    labels = os.listdir(frames_folder)
    label_len = len(labels)
    frames_folders = [frames_folder] * label_len
    dst_folders = [dst_folder] * label_len

    thread_data = zip(frames_folders, labels)
    max_workers = 10
    with futures.ThreadPoolExecutor(max_workers, worker_init) as ex:
        try:
            json_datas = ex.map(thread_run, thread_data)
        except Exception as e:
            print(e)
            sys.exit(0)
            raise e

    processes = os.cpu_count()
    with multiprocessing.Pool(processes, worker_init) as pool:
        for i in (pool.imap_unordered(process_run, zip(json_datas, dst_folders))):
            pass

def worker_init():
    # ignore the SIGINI in sub process, just print a log
    def sig_int(signal_num, frame):
        print('KeyInterrupt')
        sys.exit(0)
    signal(SIGINT, sig_int)

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('TASK',type=str,help='Task name related to inner argument.')
    parser.add_argument('JSON_MASK_PATH',type=str,help='json folder path')
    parser.add_argument('png_folder',type=str,help='target png folder path')
    args = parser.parse_args()
    return args

def main():
    # data_type = ['test', 'train']
    args = get_args()
    data_type = ['train']

    # JSON_MASK_PATH = '/home/Datasets/label_json/'+args.TASK
    # for dt in data_type:
    json_folder = args.JSON_MASK_PATH
    # args.json_folder = os.path.join('F:'+os.sep,'code','django-labeller','images','000')
    png_folder = args.png_folder

    json_folders = os.listdir(json_folder)
    with tqdm(total=len(json_folders), position=0, leave=True) as pbar:
        for idx, name in tqdm(enumerate(sorted(json_folders)[:]), position=0, leave=True):
            frames_folder = os.path.join(json_folder, name)
            dst_folder = os.path.join(png_folder, name)
            print(name)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            process(frames_folder, dst_folder)

            pbar.update()

if __name__ == '__main__':
    main()
