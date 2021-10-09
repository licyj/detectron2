from genericpath import exists
import os
import pathlib
import cv2
import numpy as np
import multiprocessing as mp
import argparse
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
'''
Program:
        This program is for visualize inference result on frames
History:
        2021/08/27 Eric Chen First release
        2021/08/29 Eric Chen 'change to function call'
        2021/08/31 Eric Cjen 'adjust code structure to argparse'
'''
# https://github.com/TannerGilbert/Detectron2-Train-a-Instance-Segmentation-Model#4-registering-the-data-set

def dataset_register(args):
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("train_dataset", {}, "{}".format(args.train_json), args.root_frame)


def cfg_setting(args, cfg):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.DATALOADER.NUM_WORKERS = 8
    #https://github.com/facebookresearch/detectron2/issues/80#issuecomment-544228514


def get_mask(predictor, imagePath, mask_folder):
    img = cv2.imread(imagePath)
    outputs = predictor(img)
    # print(outputs)
    v = Visualizer(img[:, :, ::-1],
                metadata = MetadataCatalog.get("train_dataset"),
                scale = 1,
                instance_mode = ColorMode.IMAGE_BW
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    #https://blog.csdn.net/qq_48019718/article/details/119084442
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="path/to/model")
    parser.add_argument("root_frame", type=str, help="path/to/frame/path")
    parser.add_argument("root_mask", type=str, help="path/to/target/mask/path")
    parser.add_argument("train_json", type=str, help="path/to/train_json")
    args = parser.parse_args()
    return args


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    mp.set_start_method("spawn", force=True)
    cfg = get_cfg()
    args = get_args()
    cfg_setting(args, cfg)
    dataset_register(args)
    predictor = DefaultPredictor(cfg)
    pathlib.Path(args.root_mask).mkdir(exist_ok=True)
    for video in os.listdir(args.root_frame):
        for image in os.listdir(os.path.join(args.root_frame, video)):
            imagePath = os.path.join(args.root_frame, video, image)
            get_mask(predictor, imagePath, mask_folder=args.root_mask)

if __name__ == '__main__':
    main()