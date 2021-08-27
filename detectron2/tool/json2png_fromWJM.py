import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent import futures
from signal import *
import time

def thread_run(data):
    frames_folder, label = data
    with open(os.path.join(frames_folder, label)) as f:
        json_data = json.load(f)
    return json_data

def process_run(data):
    json_data, dst_folder = data
    label_order = ['head', 'right_hand', 'left_hand', 'others']
    mask = np.zeros((720, 1280), dtype=np.uint8)
    for labels in json_data['labels']:
        filename = json_data['image_filename'].replace('.jpg', '.png')
        print(filename)

        for coord in labels['regions'][0]:
            try:
                # translate class to number of class 
                # 0:bg, 1:head, 2:right_hand, 3:left_hand
                mask[int(coord['y']), int(coord['x'])] = label_order.index(labels['label_class']) + 1
            except Exception as e:
                print(os.path.join(dst_folder, filename))
                raise e
            
        cv2.imwrite(os.path.join(dst_folder, filename), mask)
        # assert False


    # print([len(xs) for xs in mask_xs])
    # print([len(ys) for ys in mask_ys])
    # assert False

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
    # processes = 1
    with multiprocessing.Pool(processes, worker_init) as pool:
        for i in (pool.imap_unordered(process_run, zip(json_datas, dst_folders))):
            pass

def worker_init():
    # ignore the SIGINI in sub process, just print a log
    def sig_int(signal_num, frame):
        print('KeyInterrupt')
        sys.exit(0)
    signal(SIGINT, sig_int)


def main():
    # data_type = ['test', 'train']
    data_type = ['train']
    JSON_MASK_PATH = 'Datasets/TSL/mask/json_mask/'
    for dt in data_type:
        json_folder = os.path.join(JSON_MASK_PATH, dt)
        # json_folder = os.path.join('F:'+os.sep,'code','django-labeller','images','000')
        yaml_folder = os.path.join('process', dt, 'mask_png')

        json_folders = os.listdir(json_folder)
        with tqdm(total=len(json_folders), position=0, leave=True) as pbar:
            for idx, name in tqdm(enumerate(sorted(json_folders)[63:]), position=0, leave=True):
                frames_folder = os.path.join(json_folder, name)
                dst_folder = os.path.join(yaml_folder, name)
                print(name)
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)


                process(frames_folder, dst_folder)

                pbar.update()
                # assert False

        # processes = os.cpu_count()
        # with multiprocessing.Pool(processes, worker_init) as pool:
        # 	frames_folders = []
        # 	dst_folders = []
        # 	for name in os.listdir(json_folder):
        # 		frames_folders.append(os.path.join(json_folder, name))
        # 		dst_folders.append(os.path.join(yaml_folder, name))

        # 	process_data = zip(frames_folders, dst_folders)
        # 	with tqdm(total=len(frames_folders), position=0, leave=False) as pbar:
        # 		for idx, _ in enumerate(pool.imap_unordered(process, process_data)):
        # 			pbar.update()
        # for name in os.listdir(json_folder):
        # 	frames_folder = os.path.join(json_folder, name)
        # 	dst_folder = os.path.join(yaml_folder, name)
        # 	if not os.path.exists(dst_folder):
        # 		os.makedirs(dst_folder)
        # 	for label in os.listdir(frames_folder):
        # 		with open(os.path.join(frames_folder, label)) as f:
        # 			data = json.load(f)

        # 		masks = []
        # 		mask_xs = []
        # 		mask_ys = []
        # 		for labels in data['labels']:
        # 			mask = []
        # 			mask_x = []
        # 			mask_y = []
        # 			for coord in labels['regions'][0]:
        # 				mask.append([coord['x'], coord['y']])
        # 				mask_x.append(coord['x'])
        # 				mask_y.append(coord['y'])
        # 			masks.append(mask)
        # 			mask_xs.append(mask_x)
        # 			mask_ys.append(mask_y)

        # 		masks = sorted(masks, key=lambda x: len(x), reverse=True)
        # 		masks = [masks[0], masks[2], masks[1]] if len(masks) > 2 and \
        # 												min(mask_xs[1]) > min(mask_xs[2]) \
        # 				else masks

        # 		yaml_data = {}
        # 		for idx, mask in enumerate(masks):
        # 			yaml_data[label_order[idx]] = mask

        # 		# print(yaml_data)
        # 		# print(os.path.join(dst_folder, label)+'.yaml')

        # 		with open(os.path.join(dst_folder, label)+'.yaml', 'w') as f:
        # 			yaml.dump(yaml_data, f)

        # 		# if len(masks) > 1:
        # 		# 	print(os.path.join(dst_folder, label)+'.yaml')
        # 		# 	mask = yaml.load(open(os.path.join(dst_folder, label)+'.yaml', 'r'), Loader=yaml.FullLoader)
        # 		# 	print(mask)
        # 		# 	assert False

if __name__ == '__main__':
    main()

