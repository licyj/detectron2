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
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results
from Trainer import Trainer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
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
    for train_set in cfg.DATASETS.TRAIN:
        if train_set is 'TSL':
            register_coco_instances(train_set, {}, \
                '/home/Datasets/TSL/train/train.json',\
                '/home/Datasets/TSL/train/frames/')    
            MetadataCatalog.get(train_set)
        else:
            register_coco_instances(train_set, {}, \
                '/home/Datasets/ASL/train/{}/train_{}.json'.format(train_set, train_set),\
                '/home/Datasets/ASL/train/{}/frame/'.format(train_set))
            MetadataCatalog.get(train_set)




    for val_set in cfg.DATASETS.TEST:
        register_coco_instances(val_set, {}, \
            '/home/Datasets/ASL/train/{}/val_{}.json'.format(val_set,val_set),\
            '/home/Datasets/ASL/train/{}/val/'.format(val_set))
        MetadataCatalog.get(val_set)


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    cfg = setup(args)
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