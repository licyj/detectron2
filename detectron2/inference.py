#coding='utd-8'
#https://blog.csdn.net/qq_48019718/article/details/118859447

import os, random, cv2, time, json
from posixpath import dirname
import numpy as np
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
'''
Program:
        This program is for visualize inference result on frames
        Output videos finally
History:
        2021/08/27 Eric Chen First release
Usage:
        python inference.py
'''
CUSTOM_TEST_JSON_PATH = os.path.join('/home/Datasets/','all_videos_json')
OUTPUT_MASK_OATH = '/home/Datasets/mask/'
MODEL_DIR = os.path.join('output')
TASK = 'ASL_test-all-videos'


# =======json format translate frome labelme =======
# https://github.com/TannerGilbert/Detectron2-Train-a-Instance-Segmentation-Model#4-registering-the-data-set
def get_test_dicts(directory):
    '''
        input: CUSTOM_TEST_JSON_PATH
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
            # TODO: note path is correct 
            filename = img_anns["imagePath"]
            record["file_name"] = filename
            img = cv2.imread(filename)
            h,w = img.shape[:2]
            record["height"] = h
            record["width"] = w
            # annos = img_anns["shapes"]
            # objs = []
            # for anno in annos:
            #     px = [a[0] for a in anno['points']]
            #     py = [a[1] for a in anno['points']]
            #     poly = [(x, y) for x, y in zip(px, py)]
            #     poly = [p for x in poly for p in x]

            #     obj = {
            #         "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            #         "bbox_mode": BoxMode.XYXY_ABS,
            #         "segmentation": [poly],
            #         "category_id": classes.index(anno['label']),
            #         "iscrowd": 0
            #     }
            #     objs.append(obj)
            record["annotations"] = []
            dataset_dicts.append(record)
    return dataset_dicts



###########################    ASL   register  #########################
#https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-dataset
DatasetCatalog.register(TASK, get_test_dicts)
ASL_train_meta = MetadataCatalog.get(TASK)


############################  wjm test register ###########################
# for d in ["train"]:
#     register_coco_instances(f"TSL_{d}", {}, \
#                             f"TSL_segmentation/{d}/{d}.json", \
#                             f"TSL_segmentation/{d}/frames/")


#################### inference 5 video labeled before. ####################
# test_set_dicts = DatasetCatalog.get(TASK)
# TSL_metadata = MetadataCatalog.get(TASK).set(thing_classes=["head","right_hand","left_hand"])


#############################  config setting #############################
cfg = get_cfg()
# new
cfg.merge_from_file(
    "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIR, "model_final_TSL_754.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
# new 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.DATASETS.TEST = (TASK, )
predictor = DefaultPredictor(cfg)
#https://github.com/facebookresearch/detectron2/issues/80#issuecomment-544228514
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
# ASL_test_meta = MetadataCatalog.get("ASL_test")
mask_folder = os.path.join( OUTPUT_MASK_OATH,\
            "{}_{}".format(cfg.MODEL.WEIGHTS.split("/")[-1].replace(".pth",""),
                        '0827'))
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

test_set_dicts = get_test_dicts(CUSTOM_TEST_JSON_PATH)
for d in test_set_dicts:    
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                   metadata = ASL_train_meta,
                   scale = 1,
                   instance_mode = ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # new
    img = v.get_image()[:, :, ::-1]
    # cv2.imwrite(TEST_IMG_PATH+'{}.jpg'.format(d["file_name"]), img)
    # save bbox 
    
    #TODO: get mask and save
    #https://blog.csdn.net/qq_48019718/article/details/119084442
    mask_array = outputs['instances'].to("cpu").pred_masks.numpy()
    # print("zeros:",np.count_nonzero(mask_array))
    num_instances = mask_array.shape[0]         #有几个目标
    mask_array = np.moveaxis(mask_array, 0, -1) #移动 shape 的尺寸
    mask_array_instance = []
    output = np.zeros_like(img)
    num,h,width = mask_array.shape
    for i in range(num_instances):
        mask_array_instance.append(mask_array[:, :, i:(i+1)])
        output = np.where(mask_array_instance[i] == True,i+1, output)
        # output = np.where(mask_array_instance[i] == True,255, output)
    video_folder = os.path.join(mask_folder,d["file_name"].split('/')[-2])
    if not os.path.exists(video_folder):
        os.mkdir(video_folder)
    mask_path = os.path.join(video_folder, d["file_name"].split('/')[-1])
    cv2.imwrite(mask_path,output)