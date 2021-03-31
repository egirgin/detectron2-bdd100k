import logging
import os
import shutil
from collections import OrderedDict
from tqdm import tqdm
import random
import cv2

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    DatasetMapper,
    DatasetCatalog,
)

from utils.dataloader import mapper as custom_mapper

from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import ColorMode


logger = logging.getLogger("detectron2")

from utils.dataloader import get_dataset_dicts

# TODO: checkpointer, rms loss, LR scheduler, tensorboard accuracy printing, resume (both weights and the dataset),


class MyTrainer:

    def __init__(self, experiment_name, eval_period = 5, checkpoint_period = 100):
        self.testset = None
        self.cfg = None
        self.model = None
        self.eval_period = eval_period
        self.checkpoint_period = checkpoint_period
        self.name = experiment_name

    def build_model(self, cfg):

        self.cfg = cfg

        self.model = build_model(cfg)

        return self.model

    def set_datasets(self, trainset=None, valset=None, testset=None):
        if testset:
            self.testset = testset

        if trainset:
            self.cfg.DATASETS.TRAIN = (trainset,)

        if valset:
            self.cfg.DATASETS.TEST = (valset,)

        self.build_model(self.cfg)

    def train(self, epochs, batch_size, resume=False):
        global start_iter
        if self.model:
            # Switch to training mode
            self.model.train()

            # Build Optimizer (SGD)
            optimizer = build_optimizer(self.cfg, self.model)  # Returns SGD
            # optimizer = torch.optim.Adam()

            scheduler = build_lr_scheduler(self.cfg, optimizer) # warm up scheduler

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
                                           output_dir=self.cfg.OUTPUT_DIR + "/eval")

            writers = default_writers(self.cfg.OUTPUT_DIR, train_iter)


            checkpointer = DetectionCheckpointer(
                self.model, self.cfg.OUTPUT_DIR + "/checkpoint", optimizer=optimizer, scheduler=scheduler
            )
            if resume:
                if checkpointer.has_checkpoint():
                    latest_str = checkpointer.get_checkpoint_file()
                    print("Loading checkpoint: {}".format(latest_str))
                    load = checkpointer.resume_or_load(latest_str, resume=resume)
                    print(load.keys())
                    start_iter = load["iteration"] + 1
                    print("Starting Iteration : {}".format(start_iter))
                else:
                    print("Error! There is not any checkpoints. Training from scratch!")

            else:
                start_iter = 0

            periodic_checkpointer = PeriodicCheckpointer(
                checkpointer, self.checkpoint_period*train_size,
                max_iter=train_iter, max_to_keep=3, file_prefix=self.name
            )

            with EventStorage(start_iter=start_iter) as storage:
                for data, iteration in zip(train_loader, range(start_iter, train_iter)):
                    print("One epoch takes {} iterations.".format(train_size))
                    loss_dict = self.model(data)

                    # squared = torch.FloatTensor([torch.square(x) for x in list(loss_dict.values())])

                    # losses = torch.sqrt(torch.mean(squared))

                    losses = sum(loss_dict.values())

                    for key, value in loss_dict.items():
                        storage.put_scalar(key, value / batch_size)

                    storage.put_scalar("RMSloss", losses)

                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"])

                    assert torch.isfinite(losses).all(), loss_dict

                    optimizer.zero_grad()

                    losses.backward()

                    optimizer.step()


                    if (iteration) % (self.eval_period * train_size) == 0:
                        eval_acc, eval_loss = self.eval(acc_loader, val_loader)

                        print(eval_acc)
                        print(eval_loss)

                        storage.put_scalar("Accuracy/bbox_mAP", eval_acc["bbox"]["AP"])
                        storage.put_scalar("Accuracy/bbox_mAP_main_pth", eval_acc["bbox"]["AP-main_path"])
                        storage.put_scalar("Accuracy/bbox_mAP_alt_pth", eval_acc["bbox"]["AP-alt_path"])

                        storage.put_scalar("Accuracy/segm_mAP", eval_acc["segm"]["AP"])
                        storage.put_scalar("Accuracy/segm_mAP_main_pth", eval_acc["segm"]["AP-main_path"])
                        storage.put_scalar("Accuracy/segm_mAP_alt_pth", eval_acc["segm"]["AP-alt_path"])

                        for key, value in eval_loss.items():
                            storage.put_scalar("ValLoss/{}".format(key), value / val_size)

                    storage.step()

                    for writer in writers:
                        writer.write()

                    periodic_checkpointer.step(iteration)
                    scheduler.step()

        else:
            print("Please build the model first!")

    def eval(self, acc_loader, val_loader):

        eval_results = inference_on_dataset(
            self.model,
            acc_loader,
            self.evaluator)

        total_loss = {}

        # self.model.eval()
        with torch.no_grad():
            for val in val_loader:
                loss_dict = self.model(val)

                total_loss = {k: total_loss.get(k, 0) + loss_dict.get(k, 0) for k in set(total_loss) | set(loss_dict)}

        # self.model.train()

        return eval_results, total_loss

    def test(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "checkpoint/model_final.pth")  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

        predictor = DefaultPredictor(self.cfg)

        for img in random.sample(os.listdir(self.testset), 3):
            im = cv2.imread(self.testset + "/" + img)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=MetadataCatalog.get("train"),
                           scale=0.5,
                           instance_mode=ColorMode.IMAGE_BW
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            os.makedirs(self.cfg.OUTPUT_DIR + "/results", exist_ok=True)

            cv2.imwrite(self.cfg.OUTPUT_DIR + "/results/{}".format(img), out.get_image()[:, :, ::-1])
            try:
                cv2.imshow(out.get_image()[:, :, ::-1])
            except:
                pass




        pass
