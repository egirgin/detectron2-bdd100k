import logging
import os
import math

import random
import cv2
from PIL import Image
import numpy as np

import torch

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,

)

from utils.dataloader import mapper as custom_mapper

from detectron2.engine import default_writers
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import ColorMode

logger = logging.getLogger("detectron2")


class MyTrainer:

    def __init__(self, experiment_name, eval_period=5, checkpoint_period=100):
        self.cfg = None
        self.model = None
        self.eval_period = eval_period
        self.checkpoint_period = checkpoint_period
        self.name = experiment_name

    def build_model(self, cfg):

        self.cfg = cfg

        self.model = build_model(cfg)

        return self.model

    def set_datasets(self, trainset=None, valset=None):
        if trainset:
            self.cfg.DATASETS.TRAIN = (trainset,)

        if valset:
            self.cfg.DATASETS.TEST = (valset,)

        self.build_model(self.cfg)

    def train(self, epochs, batch_size, resume=False):
        if self.model:
            # Switch to training mode
            self.model.train()

            # Build Optimizer (SGD)
            optimizer = build_optimizer(self.cfg, self.model)  # Returns SGD
            # optimizer = torch.optim.Adam()

            scheduler = build_lr_scheduler(self.cfg, optimizer)  # warm up scheduler

            train_set = get_detection_dataset_dicts("train", filter_empty=False)

            val_set = get_detection_dataset_dicts("val", filter_empty=False)

            train_size = len(train_set)

            train_iter = train_size * epochs

            val_size = len(val_set)

            train_loader = build_detection_train_loader(train_set, mapper=custom_mapper, aspect_ratio_grouping=False,
                                                        total_batch_size=batch_size)
            val_loader = build_detection_test_loader(val_set, mapper=custom_mapper)

            acc_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

            self.evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], ("bbox", "segm",), False,
                                           output_dir=self.cfg.OUTPUT_DIR + "/" + self.name + "/eval")

            writers = default_writers(self.cfg.OUTPUT_DIR + "/" + self.name, train_iter)

            checkpointer = DetectionCheckpointer(
                self.model, self.cfg.OUTPUT_DIR + "/" + self.name + "/checkpoint", optimizer=optimizer,
                scheduler=scheduler
            )
            if resume:
                if checkpointer.has_checkpoint():
                    latest_str = checkpointer.get_checkpoint_file()
                    print("Loading checkpoint: {}".format(latest_str))
                    load = checkpointer.resume_or_load(latest_str, resume=resume)
                    start_iter = load["iteration"] + 1
                    print("Starting Iteration : {}".format(start_iter))
                else:
                    print("Error! There is not any checkpoints. Training from scratch!")

            else:
                print("Training from scratch with random initialization!")
                start_iter = 0

            periodic_checkpointer = PeriodicCheckpointer(
                checkpointer, self.checkpoint_period * train_size,
                max_iter=train_iter, max_to_keep=3, file_prefix=self.name
            )
            print("One epoch takes {} iterations.".format(train_size))
            with EventStorage(start_iter=start_iter) as storage:
                for data, iteration in zip(train_loader, range(start_iter, train_iter)):

                    loss_dict = self.model(data)

                    losses = sum([loss**2 for loss in loss_dict.values()])/len(loss_dict.values()) #RMS Loss

                    # squared = torch.FloatTensor([torch.square(x) for x in list(loss_dict.values())])

                    # losses = torch.sqrt(torch.mean(squared))

                    # losses = sum(loss_dict.values()) # ABS Loss

                    for key, value in loss_dict.items():
                        storage.put_scalar("TrainLoss/{}".format(key.upper()), value / batch_size)

                    storage.put_scalar("RMSLoss/Trainloss", losses)

                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"])

                    assert torch.isfinite(losses).all(), loss_dict

                    optimizer.zero_grad()

                    losses.backward()

                    optimizer.step()

                    periodic_checkpointer.step(iteration)
                    scheduler.step()

                    if (iteration != 0) and (iteration % (self.eval_period * train_size) == 0):
                        eval_acc, eval_loss, min_img = self.eval(acc_loader, val_loader)

                        print(eval_acc)
                        print(eval_loss)

                        storage.put_scalar("Accuracy/bbox_mAP", eval_acc["bbox"]["AP"])
                        storage.put_scalar("Accuracy/bbox_mAP_main_pth", eval_acc["bbox"]["AP-main_path"])
                        storage.put_scalar("Accuracy/bbox_mAP_alt_pth", eval_acc["bbox"]["AP-alt_path"])

                        storage.put_scalar("Accuracy/segm_mAP", eval_acc["segm"]["AP"])
                        storage.put_scalar("Accuracy/segm_mAP_main_pth", eval_acc["segm"]["AP-main_path"])
                        storage.put_scalar("Accuracy/segm_mAP_alt_pth", eval_acc["segm"]["AP-alt_path"])

                        for key, value in eval_loss.items():
                            storage.put_scalar("ValLoss/{}".format(key.upper()), value / val_size)

                        val_rms = sum([loss**2 for loss in eval_loss.values()])/len(eval_loss.values())

                        storage.put_scalar("RMSLoss/Valloss", val_rms)

                        if checkpointer.has_checkpoint():
                            latest_str = checkpointer.get_checkpoint_file()
                            print("Loading checkpoint: {}".format(latest_str))
                            self.cfg.MODEL.WEIGHTS = latest_str
                            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                            predictor = DefaultPredictor(self.cfg)

                            min_img = min_img.permute(1, 2, 0)

                            min_img = min_img.detach().cpu().numpy()

                            outputs = predictor(min_img)
                            if len(outputs["instances"].get_fields()["pred_boxes"]) == 0:
                                print("No instances found :(")
                                continue
                            v = Visualizer(min_img[:, :, ::-1],
                                           metadata=MetadataCatalog.get("val"),
                                           scale=0.5,
                                           instance_mode=ColorMode.IMAGE_BW
                                           )
                            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                            img = np.transpose(out.get_image(), (2, 0, 1))

                            cv2.imwrite(
                                self.cfg.OUTPUT_DIR + "/" + self.name + "/prediction_{}.png".format(iteration),
                                img)

                            storage.put_image("Iter:{}".format(iteration), img)
                        else:
                            print("No checkpoint found! Skipping this epoch.")

                    storage.step()

                    for writer in writers:
                        writer.write()



        else:
            print("Please build the model first!")

    def eval(self, acc_loader, val_loader):

        eval_results = inference_on_dataset(
            self.model,
            acc_loader,
            self.evaluator)

        total_loss = {}
        min_loss = 999999999

        # self.model.eval()
        with torch.no_grad():
            for val in val_loader:
                loss_dict = self.model(val)

                if sum(loss_dict.values()) < min_loss:
                    min_img = val[0]["image"]

                total_loss = {k: total_loss.get(k, 0) + loss_dict.get(k, 0) for k in set(total_loss) | set(loss_dict)}

        # self.model.train()

        return eval_results, total_loss, min_img
