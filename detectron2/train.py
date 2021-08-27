# github: 
# Object-Detection-and-Image-Segmentation-with-Detectron2
# https://bit.ly/3syJCcC
# #nice to reference: https://bit.ly/3jjU4lz
# ############################  common lib ###################################
import torch, torchvision 
assert torch.__version__.startswith("1.7")
print(torch.__version__, torch.cuda.is_available())
import detectron2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random, json
from time import gmtime, strftime
# ################# import some common detectron2 utilities #################
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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Before training, it need to register dataset first.
# option 1:
################################### Custom dataset register ####################################################
# https://gilberttanner.com/blog/detectron2-train-a-instance-segmentation-model
'''
def get_ASL_dicts(directory):
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

for d in ["train", "test"]:
    DatasetCatalog.register("ASL_" + d, lambda d=d: get_ASL_dicts('ASL_Segmentation/' + d+"_imges"))
    MetadataCatalog.get("ASL_" + d).set(thing_classes=['head', 'left_hand', 'right_hand'])
ASL_metadata = MetadataCatalog.get("ASL_train")
'''


# option 2:
################################### COCO dataset register ####################################################
## train ASL
# for d in ["train", "test"]:
#     register_coco_instances(f"ASL_{d}", {}, f"ASL_segmentation/{d}/{d}.json", f"ASL_segmentation/{d}/{d}_images")

## train TSL
# for d in ["train", "test"]:

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
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

'''
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)
'''


TASK = 'ASL_train-1000'
###################################  TSL register  ###################################
# for d in ["train"]:
#     register_coco_instances(TASK, {},\
#                             f"TSL_segmentation/{d}/{d}.json",\
#                             f"TSL_segmentation/{d}/frames/")

################################## ASL register #####################################
for d in ["train"]:
    register_coco_instances(TASK, {},\
                            f"Datasets/ASL/train/trainset/{d}.json",\
                            f"Datasets/ASL/train/trainset/image/")

################################### config setting  ###################################
cfg = get_cfg()
cfg.DATASETS.TRAIN = (TASK,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# pre-trained model usage
# cfg.merge_from_file('output/model_final_TSL_754.pth')
cfg.MODEL.WEIGHTS = 'output/model_final_TSL_754.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025
# setting cfg.SOLVER.MAX_ITER
# https://stackoverflow.com/questions/63578040/how-many-images-per-iteration-in-detectron2
cfg.SOLVER.MAX_ITER = 10000 # train longer 
# time: https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
cfg.OUTPUT_DIR = 'runs/'+'{}-iter:{}-batch:{}-{}'.format(TASK,\
                                                cfg.SOLVER.MAX_ITER,\
                                                cfg.SOLVER.IMS_PER_BATCH,\
                                                strftime("%Y-%m-%d %H:%M:%S", gmtime()))


################################### data augmentation ###################################
# augs = T.AugmentationList([
#     T.RandomBrightness(0.9,1.1),
#     T.RandomFlip(prob=0.5),
#     # T.RandomCrop("absolute",(720,720))
# ])


# model = build_model(cfg)  # returns a torch.nn.Module
if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

################################### training ###################################
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



'''
# faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  
'''
