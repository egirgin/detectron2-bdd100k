import logging
import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

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

from utils.dataloader import get_dataset_dicts


# TODO: checkpointer, rms loss, LR scheduler, tensorboard accuracy printing, resume (both weights and the dataset),

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

    def train(self, epochs, batch_size, resume=False):
        if self.model:
            # Switch to training mode
            self.model.train()

            # Build Optimizer (SGD)
            optimizer = build_optimizer(self.cfg, self.model)  # Returns SGD
            # optimizer = torch.optim.Adam()

            # scheduler = build_lr_scheduler(self.cfg, optimizer) # warm up scheduler

            checkpointer = DetectionCheckpointer(
                self.model, self.cfg.OUTPUT_DIR + "/checkpoint", optimizer=optimizer
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
                checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=epochs, max_to_keep=3, file_prefix="test"
            )

            train_set = get_detection_dataset_dicts("train", filter_empty=False)

            val_set = get_detection_dataset_dicts("val", filter_empty=False)

            print(train_set[0].keys())

            train_loader = build_detection_train_loader(train_set, mapper=custom_mapper, aspect_ratio_grouping=False,
                                                        total_batch_size=batch_size)
            val_loader = build_detection_test_loader(val_set, mapper=custom_mapper)


            self.evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], ("segm"), False,
                                           output_dir=self.cfg.OUTPUT_DIR + "/eval")

            writers = default_writers(self.cfg.OUTPUT_DIR, epochs)

            with EventStorage(start_iter=start_iter) as storage:
                for data, iteration in zip(train_loader, range(start_iter, epochs)):

                    loss_dict = self.model(data)

                    # squared = torch.FloatTensor([torch.square(x) for x in list(loss_dict.values())])

                    # losses = torch.sqrt(torch.mean(squared))

                    losses = sum(loss_dict.values())

                    for key, value in loss_dict.items():
                        storage.put_scalar(key, value)

                    storage.put_scalar("RMSloss", losses)

                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"])

                    assert torch.isfinite(losses).all(), loss_dict

                    optimizer.zero_grad()

                    losses.backward()

                    optimizer.step()

                    if (iteration + 1) % 5 == 0:
                        eval_acc, eval_loss = self.eval(val_loader)

                        print(eval_acc)
                        print(eval_loss)

                        storage.put_scalar("Accuracy/bbox_mAP", eval_acc["bbox"]["AP"])
                        storage.put_scalar("Accuracy/bbox_mAP_main_pth", eval_acc["bbox"]["AP-main_path"])
                        storage.put_scalar("Accuracy/bbox_mAP_alt_pth", eval_acc["bbox"]["AP-alt_path"])

                        storage.put_scalar("Accuracy/segm_mAP", eval_acc["segm"]["AP"])
                        storage.put_scalar("Accuracy/segm_mAP_main_pth", eval_acc["segm"]["AP-main_path"])
                        storage.put_scalar("Accuracy/segm_mAP_alt_pth", eval_acc["segm"]["AP-alt_path"])

                    storage.step()

                    for writer in writers:
                        writer.write()

                    periodic_checkpointer.step(iteration)

        else:
            print("Please build the model first!")
            
    


    def eval(self, val_loader):

        eval_results = inference_on_dataset(
            self.model,
            val_loader,
            self.evaluator)

        total_loss = {}

        self.model.eval()
        with torch.no_grad():
            for val in val_loader:
                loss_dict = self.model(val)

                total_loss = {k: total_loss.get(k, 0) + loss_dict.get(k, 0) for k in set(total_loss) | set(loss_dict)}

        self.model.train()

        return eval_results, total_loss

    def test(self):

        pass
