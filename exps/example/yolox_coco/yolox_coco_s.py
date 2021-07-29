# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir, ConcatDataset, MixConcatDataset


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 3

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.train_ann = "train.json"
        self.val_ann = "train.json"   # 这里也是train.json
        self.yolox_datadir = "/path/to/your/datasets"  # /COCO
        self.data_train_dirs = ["/home/llsq/DATA/myCoco/ball/basketball/gx_v2"
                                "/home/llsq/DATA/myCoco/ball/basketball/gx_v3",
                                "/home/llsq/DATA/myCoco/ball/basketball/gx_v4",
                                # "/home/llsq/DATA/myCoco/ball/basketball/gx_v5",
                                # "/home/llsq/DATA/myCoco/ball/basketball/gx_video1"
                                ]
        self.data_val_dirs = ["/home/llsq/DATA/myCoco/ball/basketball/gx_video1"]
        self.data_test_dirs = ["/home/llsq/DATA/myCoco/ball/basketball/gx_video1"]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        data_sets = []
        for train_data_dir in self.data_train_dirs:
            dataset = COCODataset(
                data_dir=train_data_dir,
                json_file=self.train_ann,
                name="images",
                img_size=self.input_size,
                preproc=TrainTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_labels=50,
                ),
            )

            dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_labels=120,
                ),
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                enable_mixup=self.enable_mixup,
            )
            # self.dataset = dataset
            data_sets.append(dataset)

        self.dataset = MixConcatDataset(data_sets)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler}
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import COCODataset, ValTransform

        data_val_sets = []
        for val_data_dir in self.data_val_dirs:
            valdataset = COCODataset(
                data_dir=val_data_dir,
                json_file=self.val_ann,  # if not testdev else "image_info_test-dev2017.json",
                name="images",  # if not testdev else "test2017",
                img_size=self.test_size,
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            )
            data_val_sets.append(valdataset)
        valdatasets = ConcatDataset(data_val_sets)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdatasets, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdatasets)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler,
                             "batch_size": batch_size}
        val_loader = torch.utils.data.DataLoader(valdatasets, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
