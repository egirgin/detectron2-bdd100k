import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
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
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")


class MyTrainer:

    def __init__(self):
        self.testset = None
        self.cfg = None
        self.model = None

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

    def train(self, epochs):
        if self.model:
            # Switch to training mode
            self.model.train()

            # Build Optimizer (SGD)
            optimizer = build_optimizer(self.cfg, self.model) # Returns SGD
            #optimizer = torch.optim.Adam()

            #scheduler = build_lr_scheduler(self.cfg, optimizer) # warm up scheduler

            train_loader = build_detection_train_loader(self.cfg)

            with EventStorage(start_iter=0) as storage:
                for data, iteration in zip(train_loader, range(0, epochs)):
                    storage.iter = iteration

                    loss_dict = self.model(data)

                    losses = sum(loss_dict.values())

                    print("Iteration: {}, Loss: {}".format(iteration, losses))

                    assert torch.isfinite(losses).all(), loss_dict

                    optimizer.zero_grad()

                    losses.backward()

                    optimizer.step()







        else:
            print("Please build the model first!")

    def eval(self):
        pass

    def test(self):
        pass

