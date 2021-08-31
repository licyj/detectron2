import os
import random
import cv2
import time
import json
from posixpath import dirname
import numpy as np
import argparse
from time import gmtime, strftime
import matplotlib  as plt
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
#https://blog.csdn.net/qq_48019718/article/details/118859447
'''
Program:
        This program is for visualize inference result on frames
        Output videos finally
History:
        2021/08/27 Eric Chen First release
        2021/08/29 Eric Chen 'change to function call'
        2021/08/31 Eric Cjen 'adjust code structure to argparse'
Usage:
        python inference.py
'''
### need change ###
# TASK = '0831valid_video'
# CUSTOM_TEST_JSON_PATH = os.path.join('/home/Datasets/','ASL/train/0829valid_video/json/spreadthesign')
###################

# https://github.com/TannerGilbert/Detectron2-Train-a-Instance-Segmentation-Model#4-registering-the-data-set
def get_test_dicts(directory):
    '''
        input: directory of json need to create mask
        output: list of image entry
    '''
    classes = ['head', 'right_hand', 'left_hand']
    dataset_dicts = []
    for video in os.listdir(directory):
        for jsonfile in os.listdir(directory+'/'+video):
            JSONFILE = os.path.join(directory,video,jsonfile)
            with open(JSONFILE,'r') as f:
                img_anns = json.load(f)
            record = {}
            filename = img_anns["imagePath"]
            record["file_name"] = filename
            img = cv2.imread(filename)
            h,w = img.shape[:2]
            record["height"] = h
            record["width"] = w
            if 'shapes' in img_anns.keys():
                annos = img_anns["shapes"]
                objs = []
                for anno in annos:
                    px = [a[0] for a in anno['points']]
                    py = [a[1] for a in anno['points']]
                    poly = [(x, y) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": classes.index(anno['label']),
                        "iscrowd": 0
                    }
                    objs.append(obj)
            record["annotations"] = []
            dataset_dicts.append(record)
    return dataset_dicts

def dataset_register(args):
    #https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-dataset
    DatasetCatalog.register(args.TASK, get_test_dicts)
    train_meta = MetadataCatalog.get(args.TASK)

    return train_meta

def cfg_setting(args, cfg):
    cfg.MODEL.WEIGHTS = os.path.join(args.model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = (args.TASK, )
    #https://github.com/facebookresearch/detectron2/issues/80#issuecomment-544228514
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False

def get_mask(predictor, d, train_meta, mask_folder):
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                metadata = train_meta,
                scale = 1,
                instance_mode = ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    #https://blog.csdn.net/qq_48019718/article/details/119084442
    mask_array = outputs['instances'].to("cpu").pred_masks.numpy()
    num_instances = mask_array.shape[0]         #有几个目标
    mask_array = np.moveaxis(mask_array, 0, -1) #移动 shape 的尺寸
    mask_array_instance = []
    output = np.zeros_like(img)
    for i in range(num_instances):
        mask_array_instance.append(mask_array[:, :, i:(i+1)])
        output = np.where(mask_array_instance[i] == True, i+1, output)
    video_folder = os.path.join(mask_folder, d["file_name"].split('/')[-2])
    if not os.path.exists(video_folder):
        os.mkdir(video_folder)
    mask_path = os.path.join(video_folder, d["file_name"].split('/')[-1].replace('jpg','png'))
    cv2.imwrite(mask_path, output)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("TASK",
                        type=str,
                        help="task name")
    parser.add_argument("CUSTOM_TEST_JSON_PATH",
                        type=str,
                        help="json files within testset")
    parser.add_argument("model",
                        type=str,
                        help="model name")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cfg = get_cfg()
    cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg_setting(args, cfg)
    train_meta = dataset_register(args= args)
    predictor = DefaultPredictor(cfg)
    test_set_dicts = get_test_dicts(args.CUSTOM_TEST_JSON_PATH)
    OUTPUT_MASK_PATH = os.path.join('/home/Datasets/mask/', args.TASK)
    for d in test_set_dicts:    
        if not os.path.exists(OUTPUT_MASK_PATH):
            os.makedirs(OUTPUT_MASK_PATH)
        get_mask(predictor, d, train_meta, mask_folder=OUTPUT_MASK_PATH)

if __name__ == '__main__':
    main()