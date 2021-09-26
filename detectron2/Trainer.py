import os

from LossEvalHook import LossEvalHook
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.data.build import build_detection_train_loader, build_detection_test_loader
import logging
import torch
import copy
import cv2
from detectron2.modeling import GeneralizedRCNNWithTTA
from collections import OrderedDict
from detectron2.data import transforms as T 
from detectron2.data.transforms import Augmentation
from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils\
    import transform_instance_annotations,annotations_to_instances,filter_empty_instances
'''
    function:
        self defined "mapper" and "loss evaluation hook"
        implement:
            validation set loss calculation
            show validation loss on tensorboard
            real-time data augmentation
    reference:
        - train detectron2 with validation set
        https://ortegatron.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
'''

'''
def build_sem_seg_train_aug(cfg):
        function:
            implement data augmentaion
        based-on:
            https://github.com/facebookresearch/detectron2/blob/main/projects/DeepLab/train_net.py
        reference:
            official:
                tutorial:
                    https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
                github:
                    https://github.com/facebookresearch/detectron2/blob/23486b6f503490d8c526d206eb057ec33615f2de/detectron2/data/transforms/augmentation_impl.py#L399
            community:
                - https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/augmentation.md
                - https://github.com/facebookresearch/detectron2/issues/1763
                - https://github.com/facebookresearch/detectron2/issues/527
                - https://github.com/facebookresearch/detectron2/issues/1107
    
    # augs=[
    #     T.RandomApply(T.Resize(720, 1280),prob=0.5),
    #     T.RandomApply(T.RandomBrightness(0.5, 1.5),prob=0.5),
    #     T.RandomApply(T.RandomContrast(0.5, 1.5),prob=0.5),
    #     T.RandomApply(T.RandomFlip(prob=0.7, horizontal=True, vertical=False),prob=0.5),
    #     T.RandomApply(T.RandomCrop("absolute", (720,720)),prob=0.5),
    #     T.RandomApply(T.RandomRotation([10.0, 25.0], expand=True),prob=0.5)
    # ]
    augs=[
        T.Resize(720, 1280),
        T.RandomBrightness(0.5, 1.5),
        T.RandomContrast(0.5, 1.5),
        T.RandomFlip(prob=0.7, horizontal=True, vertical=False),
        T.RandomCrop("absolute", (720,720)),
        T.RandomRotation([10.0, 25.0], expand=True)
    ]
    return augs
'''

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    # Evaluator
    # https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.COCOEvaluator
    # issue: AP=0
    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/307
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
        # Eric comment on 09/11@Sat.
        # return build_evaluator(cfg, dataset_name, output_folder)


    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
                    self.cfg,
                    self.model,
                    build_detection_test_loader(
                        self.cfg, 
                        self.cfg.DATASETS.TEST[0], 
                        DatasetMapper(self.cfg,True))
                    ))
        return hooks

    # Transforms
    # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py
    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg, mapper=custom_mapper)
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                                                     T.RandomApply(T.RandomBrightness(0.5, 3),prob=0.80),
                                                     T.RandomApply(T.RandomContrast(0.5, 3),prob=0.80),
                                                     T.RandomApply(T.RandomFlip(prob=0.7, horizontal=True, vertical=False),prob=0.95),
                                                     T.RandomApply(T.RandomRotation([0.0, 360.0], expand=True),prob=0.9),
                                                     T.RandomApply(T.RandomSaturation(0.5, 3.0),prob=0.20),
                                                    # T.Resize(720, 1280), 
                                                    # T.RandomCrop("absolute", (720,720)),
                                                    ]
                                            ))

    
    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper)


    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)  
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res