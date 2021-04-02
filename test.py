import os
import random
import argparse

import cv2
import torch

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

    for img in random.sample(os.listdir(testset_path), 3):
        im = cv2.imread(testset_path + "/" + img)
        outputs = predictor(im)
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
        except:
            pass


if __name__ == '__main__':

    DatasetCatalog.register("test", lambda path="./subdataset/": get_dataset_dicts(path + "imgs/val",
                                                                              path + "bitmaps/val"))
    MetadataCatalog.get("test").set(thing_classes=["main_path", "error", "alt_path"])
    metadata_val = MetadataCatalog.get("test")

    cfg = get_cfg()

    cfg.merge_from_file("config.yaml")
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    test(cfg, args.weights, args.output, args.images)
