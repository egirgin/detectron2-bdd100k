import os
import shutil
import random
import argparse

import numpy as np
import cv2

import torch


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, SimpleTrainer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader

from utils.dataloader import get_dataset_dicts,create_subdataset

########################## ARG PARSE #################################################

argparser = argparse.ArgumentParser()

argparser.add_argument("-t", "--trainer", choices=["simple", "default", "custom"])
argparser.add_argument("-b", "--bitmap_path")
argparser.add_argument("-i", "--images_path")
argparser.add_argument("--sample", action="store_true")
argparser.add_argument("-s", "--shrink", default=0, type=int)

args = argparser.parse_args()

shrink = args.shrink

show_sample = args.sample

bitmap_path = args.bitmap_path

images_path = args.images_path

########################## SUB DATASET ###############################################

if shrink != 0:
    print("Shrinking dataset...")

    subdataset_name = "./subdataset"

    create_subdataset(images_path, bitmap_path, subdataset_name, shrink)
    bitmap_path = subdataset_name + "/bitmaps"
    images_path = subdataset_name + "/imgs"

########################## CREATE DATASET ############################################

dataset_name = "train"

try:
    DatasetCatalog.remove("train")
    DatasetCatalog.remove("val")
except:
    pass

# Trainset
print("Creating trainset...")

DatasetCatalog.register("train", lambda images_path=images_path : get_dataset_dicts(images_path + "/train", bitmap_path + "/train"))
MetadataCatalog.get("train").set(thing_classes=["main_path", "error", "alt_path"])
metadata_train = MetadataCatalog.get("train")

# Valset
print("Creating valset...")

DatasetCatalog.register("val", lambda images_path=images_path : get_dataset_dicts(images_path  + "/val", bitmap_path + "/val"))
MetadataCatalog.get("val").set(thing_classes=["main_path", "error", "alt_path"])
metadata_val = MetadataCatalog.get("val")

if show_sample:
    dataset_dicts = get_dataset_dicts(images_path + "/train", bitmap_path + "/train")
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"].split("/")[-1], out.get_image()[:, :, ::-1])

        cv2.waitKey(0)
        cv2.destroyAllWindows()

########################## CREATE MODEL ##############################################

cfg = get_cfg()

cfg.merge_from_file("config.yaml")
"""
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val")
cfg.DATALOADER.NUM_WORKERS = 1
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
"""
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if args.trainer == "simple":

    model = build_model(cfg)

    train_loader = build_detection_train_loader(cfg)

    opt = build_optimizer(cfg, model)

    trainer = SimpleTrainer(model, train_loader, opt)

    trainer.train(start_iter=0, max_iter=100)

elif args.trainer == "default":
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
elif args.trainer == "custom":
    pass









