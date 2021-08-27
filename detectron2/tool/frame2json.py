import os, sys
import json, cv2, time
import multiprocessing
from tqdm import tqdm
from signal import *
'''
Program:
          This program write image information into json file.
          So that it can be register in detectron as coco format
History:
          2021/08/27 Eric Chen First realease
Usage:
          python frame2json.py {path/to/base_frame_folder} {path/to/bese_output_json_folder}
'''

FRAME_FOLDDER = os.path.join('/home/Datasets/','all_videos_frame')
TARGET_JSON_FOLDER = os.path.join('/home/Datasets/','all_videos_json') 
if not os.path.exists(TARGET_JSON_FOLDER):
    os.mkdir(TARGET_JSON_FOLDER)

def arg_parse():
    pass

def process_run(data):
    frame_folder, dst_path = data
    for f in os.listdir(frame_folder):
        record = {}
        imgPath = os.path.join(frame_folder, f)
        record["file_name"] = imgPath
        img = cv2.imread(imgPath)
        h, w = img.shape[:2]
        record["height"] = h
        record["width"] = w
        record["annotations"] = []
        record["imagePath"] = imgPath
        
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        json_path = os.path.join(dst_path, f.replace("jpg","json"))

        with open(json_path,'w') as f:
            json.dump(record, f)

def worker_init():
    # ignore the SIGINI in sub process, just print a log
    def sig_int(signal_num, frame):
        print('KeyInterrupt')
        sys.exit(0)
    signal(SIGINT, sig_int)

def main():
    processes = os.cpu_count()
    for dt in os.listdir(FRAME_FOLDDER):
        frame_folder = os.path.join(FRAME_FOLDDER, dt)
        json_folder = os.path.join(TARGET_JSON_FOLDER, dt)
        if not os.path.exists(json_folder):
            os.mkdir(json_folder)
        with tqdm(total=len(FRAME_FOLDDER)) as pbar:
            with multiprocessing.Pool(processes, worker_init) as pool:
                frames_folder, dst_path = [], []
                frames_folder.append(os.path.join(frame_folder))
                dst_path.append(os.path.join(json_folder))
                for i in (pool.imap_unordered(process_run, zip(frames_folder, dst_path))):
                    pbar.update()

if __name__ == '__main__':
    main()


    # frame_folder: /home/Datasets/all_videos_frame/41578 
    # json_folder: /home/Datasets/all_videos_json/41578