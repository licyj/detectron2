from genericpath import exists
from posix import listdir
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import random
import cv2
import os
import argparse
import numpy as np
import json
import pathlib
import pickle
def config_setting(args, cfg):
    cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATALOADER.NUM_WORKERS = 8
    # cfg.merge_from_file("configs/ASL_men60.yaml")
    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    return cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path/to/model/weight')
    parser.add_argument('root_frame', type=str, help='path/to/frames')
    parser.add_argument('target_mask', type=str, help='path/to/output/mask/path')
    parser.add_argument('visualize', type=str, default=False, help='path/to/output/mask/path')

    args = parser.parse_args()
    return args


def register_dataset(args):
    from detectron2.data.datasets import register_coco_instances
    #TODO: maybe cause error
    task= args.root_frame.split('/')[-2]
    register_coco_instances("train_dataset", {}, "train_{}.json".format(task), args.root_frame)


def main():
    args = get_args()
    pathlib.Path(args.target_mask).mkdir(exist_ok=True)
    cfg = config_setting(args, get_cfg())
    register_dataset(args)
    predictor = DefaultPredictor(cfg)
    for video in os.listdir(args.root_frame):
        for image in os.listdir(os.path.join(args.root_frame, video)):
            imagePath = os.path.join(args.root_frame, video, image)
            img= cv2.imread(imagePath)
            outputs= predictor(img)
            v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("train_dataset"),\
                    scale=0.8, instance_mode=ColorMode.IMAGE_BW) # remove the colors of unsegmented pixels )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            mask_array = outputs['instances'].to("cpu").pred_masks.numpy()
            num_instances = mask_array.shape[0]
            mask_array = np.moveaxis(mask_array, 0, -1)
            mask_array_instance = []
            output = np.zeros_like(img)
            for i in range(num_instances):
                mask_array_instance.append(mask_array[:, :, i:(i+1)])
                output = np.where(mask_array_instance[i] == True, i+1, output)
            video_folder = os.path.join(mask_folder, imagePath.split('/')[-2])
            pathlib.Path(video_folder).mkdir(exist_ok=True)
            mask_path = os.path.join(video_folder, imagePath.split('/')[-1].replace('jpg','png'))
            cv2.imwrite(mask_path, output)

            if args.visualize:
                mask_folder = os.path.join(args.target_mask, video)
                pathlib.Path(mask_folder).mkdir(exist_ok=True)
                maskPath= os.path.join(mask_folder, image)
                cv2.imwrite(maskPath, v.get_image()[:, :, ::-1])


            # mask= outputs['instances'].get('pred_masks')
            # mask= mask.to('cpu')
            # num, h, w= mask.shape
            # bin_mask= np.zeros((h, w))
            
            # for m in mask:
            #     bin_mask+= m

            # filename= mask_folder+ image+'.png'
            # cv2.imwrite(filename, bin_mask)

if __name__ == '__main__':
    main()
