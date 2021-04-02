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

argparser = argparse.ArgumentParser()

argparser.add_argument("--weights")
argparser.add_argument("--output")
argparser.add_argument("--images")

args = argparser.parse_args()


def test(cfg, weights_path, output_path, testset_path):
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    print("Predicting...")
    for img in os.listdir(testset_path):
        im = cv2.imread(testset_path + "/" + img)
        outputs = predictor(im)
        if len(outputs["instances"].get_fields()["pred_boxes"]) == 0:
            print("No instances found :(")
            continue
        else:
            print("Instance found !!! Name: {}".format(img))
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("test"),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        os.makedirs(output_path, exist_ok=True)

        cv2.imwrite(output_path + "/{}".format(img), out.get_image()[:, :, ::-1])
        try:
            cv2.imshow(img, out.get_image()[:, :, ::-1])
        except Exception as e:
            print(e)
            cv2.imshow(img, out.get_image()[:, :, ::-1])

    print("Done...")


if __name__ == '__main__':

    shutil.rmtree(args.output, ignore_errors=True)

    print("Loading dataset configs...")
    DatasetCatalog.register("test", lambda path="./subdataset/": get_dataset_dicts(path + "imgs/val",
                                                                              path + "bitmaps/val"))
    MetadataCatalog.get("test").set(thing_classes=["main_path", "error", "alt_path"])
    metadata_val = MetadataCatalog.get("test")

    cfg = get_cfg()

    cfg.merge_from_file("config.yaml")
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    print("Done!")
    test(cfg, args.weights, args.output, args.images)
