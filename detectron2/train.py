# github: 
# Object-Detection-and-Image-Segmentation-with-Detectron2
# https://bit.ly/3syJCcC
# #nice to reference: https://bit.ly/3jjU4lz
import torch, torchvision 
assert torch.__version__.startswith("1.7")
print(torch.__version__, torch.cuda.is_available())
import os
import argparse
import datetime
import random
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
# ################# import some common detectron2 utilities #################
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
import detectron2.utils.comm as comm
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
# ########################### save checkpoint ##################################
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
setup_logger()


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, \
            cfg.INPUT.MAX_SIZE_TRAIN, \
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs

def dataset_register(args):
    register_coco_instances(args.TASK, {}, args.jsonfile_path, args.img_path)
        # "/home/Datasets/ASL/train/0831valid_video/train_spreadthesign.json",\
        # "/home/Datasets/ASL/train/0831valid_video/spreadthesign/")

def cfg_setting(args, cfg):
    cfg.DATASETS.TRAIN = (args.TASK,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    now = datetime.datetime.now()
    time = str(now.strftime("%Y-%m-%d_%H-%M-%S"))
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    # https://stackoverflow.com/questions/63578040/how-many-images-per-iteration-in-detectron2
    cfg.SOLVER.MAX_ITER = 10000 # train longer 
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # pre-trained model usage
    # cfg.merge_from_file('output/model_final_TSL_754.pth')
    cfg.OUTPUT_DIR = 'runs/'+'{}_{}'.format(args.TASK, time)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model = build_model(cfg)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("TASK",
                        type=str,
                        help="task name")
    parser.add_argument("jsonfile_path",
                        type=str,
                        help="train jsonfile_path")    
    parser.add_argument("img_path",
                        type=str,
                        help="train img_path")
    parser.add_argument("model",
                        type=str,
                        help="model name")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cfg = get_cfg()
    cfg_setting(args, cfg)
    dataset_register(args)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()