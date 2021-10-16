#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
from detectron2.data.catalog import MetadataCatalog
from detectron2 import model_zoo
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results
from Trainer import Trainer

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # Load default config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Add dataset root and dataset name key
    cfg.DATASETS.ROOT = cfg.DATASETS.NAME = None

    # Load custom config
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def dataset_register(cfg):
    """
        error may occur:
            dataset is not registered
            https://github.com/facebookresearch/detectron2/issues/253#issuecomment-550398640
    """
    for dataset_type, datasets in zip(['train', 'val'], [cfg.DATASETS.TRAIN, cfg.DATASETS.TEST]):
        for dataset in datasets:
            register_coco_instances(
                dataset,
                {},
                f'{cfg.DATASETS.ROOT}/{cfg.DATASETS.NAME}/{dataset_type}_{cfg.DATASETS.NAME}.json',
                f'{cfg.DATASETS.ROOT}/{cfg.DATASETS.NAME}/{dataset_type}/img'
            )


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print(args)
    cfg = setup_cfg(args)
    dataset_register(cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )