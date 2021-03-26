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
    DatasetCatalog
)
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

    def train(self, epochs, resume=False):
        if self.model:
            # Switch to training mode
            self.model.train()

            # Build Optimizer (SGD)
            optimizer = build_optimizer(self.cfg, self.model) # Returns SGD
            #optimizer = torch.optim.Adam()

            #scheduler = build_lr_scheduler(self.cfg, optimizer) # warm up scheduler


            checkpointer = DetectionCheckpointer(
                self.model, self.cfg.OUTPUT_DIR + "/checkpoint", optimizer = optimizer
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

            train_loader = build_detection_train_loader(self.cfg)

            val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

            val_loader_2 = build_detection_train_loader(
                self.cfg,
                dataset = DatasetCatalog.get("val")
            )

            self.evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], ("bbox", "segm"), False, output_dir=self.cfg.OUTPUT_DIR + "/eval")

            writers = default_writers(self.cfg.OUTPUT_DIR, epochs)

            with EventStorage(start_iter=start_iter) as storage:
                for data, iteration in zip(train_loader, range(start_iter, epochs)):

                    print(data[0]["instances"])
                    loss_dict = self.model(data)

                    #squared = torch.FloatTensor([torch.square(x) for x in list(loss_dict.values())])

                    #losses = torch.sqrt(torch.mean(squared))

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
                        eval_results = self.eval(val_loader, val_loader_2)

                        print(eval_results)

                        storage.put_scalar("Accuracy/bbox_mAP", eval_results["bbox"]["AP"])
                        storage.put_scalar("Accuracy/bbox_mAP_main_pth", eval_results["bbox"]["AP-main_path"])
                        storage.put_scalar("Accuracy/bbox_mAP_alt_pth", eval_results["bbox"]["AP-alt_path"])

                        storage.put_scalar("Accuracy/segm_mAP", eval_results["segm"]["AP"])
                        storage.put_scalar("Accuracy/segm_mAP_main_pth", eval_results["segm"]["AP-main_path"])
                        storage.put_scalar("Accuracy/segm_mAP_alt_pth", eval_results["segm"]["AP-alt_path"])

                    storage.step()

                    for writer in writers:
                        writer.write()

                    periodic_checkpointer.step(iteration)


        else:
            print("Please build the model first!")


    def eval(self, val_loader, val_loader_2):

        eval_results = inference_on_dataset(
            self.model,
            val_loader,
            self.evaluator)



        with torch.no_grad():
            for val_sample in tqdm(val_loader_2):
                print(val_sample[0]["instances"])

                loss_dict = self.model(val_sample)
                print(loss_dict.items())

        return eval_results

    def test(self):


        pass

