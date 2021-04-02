import os
import shutil
import random
import argparse
from datetime import datetime

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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from utils.dataloader import get_dataset_dicts, create_subdataset
from model.trainer import MyTrainer

########################## ARG PARSE #################################################

argparser = argparse.ArgumentParser()

argparser.add_argument("-n", "--name", default="")
argparser.add_argument("-t", "--trainer", choices=["simple", "default", "custom"], default="custom")
argparser.add_argument("-b", "--bitmap_path")
argparser.add_argument("-i", "--images_path")
argparser.add_argument("-e", "--epochs", type=int, default=10)
argparser.add_argument("-ep", "--eval_period", type=int, default=5)
argparser.add_argument("-bs", "--batch_size", type=int, default=4)
argparser.add_argument("-cp", "--checkpoint_period", type=int, default=1)
argparser.add_argument("--sample", action="store_true")
argparser.add_argument("--resume", action="store_true")
argparser.add_argument("-s", "--shrink", default=0, type=int)

args = argparser.parse_args()

if args.name == "":
    exp_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
else:
    exp_name = args.name

shrink = args.shrink

show_sample = args.sample

bitmap_path = args.bitmap_path

images_path = args.images_path

resume = args.resume

epochs = args.epochs
batch_size = args.batch_size
eval_period = args.eval_period

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

DatasetCatalog.register("train", lambda images_path=images_path: get_dataset_dicts(images_path + "/train",
                                                                                   bitmap_path + "/train"))
MetadataCatalog.get("train").set(thing_classes=["main_path", "error", "alt_path"])
metadata_train = MetadataCatalog.get("train")

# Valset
print("Creating valset...")

DatasetCatalog.register("val",
                        lambda images_path=images_path: get_dataset_dicts(images_path + "/val", bitmap_path + "/val"))
MetadataCatalog.get("val").set(thing_classes=["main_path", "error", "alt_path"])
metadata_val = MetadataCatalog.get("val")

if show_sample:
    dataset_dicts = get_dataset_dicts(images_path + "/val", bitmap_path + "/val")
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_val, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"].split("/")[-1], out.get_image()[:, :, ::-1])
        os.makedirs("subdataset/samples", exist_ok=True)
        cv2.imwrite("subdataset/samples/" + d["file_name"].split("/")[-1], out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

########################## CREATE MODEL ##############################################

cfg = get_cfg()

cfg.merge_from_file("config.yaml")
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

if resume:
    pass
else:
    shutil.rmtree(cfg.OUTPUT_DIR + "/" + exp_name, ignore_errors=True)

os.makedirs(cfg.OUTPUT_DIR + "/" + exp_name, exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR + "/" + exp_name + "/checkpoint", exist_ok=True)

if args.trainer == "simple":

    model = build_model(cfg)

    train_loader = build_detection_train_loader(cfg)

    opt = build_optimizer(cfg, model)

    trainer = SimpleTrainer(model, train_loader, opt)

    trainer.train(start_iter=0, max_iter=100)

elif args.trainer == "default":
    trainer = DefaultTrainer(cfg)
    trainer.build_evaluator()
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("val", ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
elif args.trainer == "custom":

    trainer = MyTrainer(eval_period=eval_period, checkpoint_period=args.checkpoint_period, experiment_name=exp_name)

    trainer.build_model(cfg)

    trainer.set_datasets(trainset="train", valset="val")

    trainer.train(epochs, batch_size, resume=resume)

