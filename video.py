import os
import random
import argparse
import shutil

import cv2
import torch
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils.dataloader import get_dataset_dicts
from utils.cam import Cam

argparser = argparse.ArgumentParser()

argparser.add_argument("--weights")
argparser.add_argument("--output")
argparser.add_argument("--images")

args = argparser.parse_args()


def predictor(cfg, weights_path):
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    return DefaultPredictor(cfg)


def run_cam(device_id=0):
    my_cam = Cam(device_id)

    my_cam.install()

    return my_cam


def read_cfg():

    print("Loading dataset configs...")
    DatasetCatalog.register("cam", lambda path="./subdataset/": get_dataset_dicts(path + "imgs/val",
                                                                                   path + "bitmaps/val"))
    MetadataCatalog.get("cam").set(thing_classes=["main_path", "error", "alt_path"])

    cfg = get_cfg()

    cfg.merge_from_file("config.yaml")
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    return cfg


if __name__ == '__main__':

    cfg = read_cfg()

    predictor = predictor(cfg, args.weights)

    my_cap = run_cam(0)

    while (True):

        ret, frame = my_cap.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        outputs = predictor(frame)
        if len(outputs["instances"].get_fields()["pred_boxes"]) == 0:
            continue
        else:
            print("{} instances found !!!".format(len(outputs["instances"].get_fields()["pred_boxes"])))

        v = Visualizer(frame[:, :, ::-1],
                       metadata=MetadataCatalog.get("test"),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imshow('cam0', out.get_image())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    my_cap.close_device()
    cv2.destroyAllWindows()




