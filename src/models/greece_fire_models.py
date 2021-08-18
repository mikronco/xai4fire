from typing import Any, List
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUC, ConfusionMatrix, AUROC
from src.models.modules.simple_dense_net import SimpleDenseNet
import torch
from torch import nn
# torch.multiprocessing.set_start_method('fork')
# torch.multiprocessing.set_start_method('fork', force=True)
# print(torch.multiprocessing.get_start_method())
import xarray as xr
import netCDF4
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import warnings
import pandas as pd
import sys
import gc
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box
from affine import Affine
import sys
import seaborn as sns
import rasterio
import os
from collections import defaultdict
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler


class LSTM_fire_model(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            hidden_size: int = 32,
            lstm_layers: int = 3,
            lr: float = 0.001,
            positive_weight: float = 0.8,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        if static_features is None:
            static_features = []
        if dynamic_features is None:
            dynamic_features = []
        self.save_hyperparameters()

        # lstm part
        self.lstm = torch.nn.LSTM(len(dynamic_features) + len(static_features), hidden_size, num_layers=lstm_layers,
                                  batch_first=True)
        # fully-connected part
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.drop1 = torch.nn.Dropout(0.5)

        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size // 2, hidden_size // 4)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc3 = torch.nn.Linear(hidden_size // 4, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.drop1,
            self.fc2,
            self.relu,
            self.drop2,
            self.fc3
        )

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        self.train_auc = AUROC(num_classes=2, pos_label=1)

        self.val_accuracy = Accuracy()
        self.val_auc = AUROC(num_classes=2, pos_label=1)

        self.test_accuracy = Accuracy()
        self.test_auc = AUROC(num_classes=2, pos_label=1)

        self.lr_scheduler_step = lr_scheduler_step
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)

    def step(self, batch: Any):
        dynamic, static, _, y = batch
        y = y.long()
        bsize, timesteps, _ = dynamic.shape
        static = static.unsqueeze(dim=1)
        repeat_list = [1 for _ in range(static.dim())]
        repeat_list[1] = timesteps
        static = static.repeat(repeat_list)

        # print(dynamic.shape, static.shape)
        inputs = torch.cat([dynamic, static], dim=2).float()
        logits = self.forward(inputs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        auc = self.val_auc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        auc = self.test_auc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_step,
                                                       gamma=self.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class CNN_fire_model(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            positive_weight: float = 0.8,
            lr: float = 0.001,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005):
        super().__init__()

        if static_features is None:
            static_features = []
        if dynamic_features is None:
            dynamic_features = []
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # CNN definition
        self.conv1 = nn.Conv2d(len(static_features) + len(dynamic_features), 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.5)
        assert (positive_weight < 1) and (positive_weight > 0)
        self.positive_weight = positive_weight

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        self.train_auc = AUROC(pos_label=1)

        self.val_accuracy = Accuracy()
        self.val_auc = AUROC(pos_label=1)

        self.test_accuracy = Accuracy()
        self.test_auc = AUROC(pos_label=1)

        self.lr_scheduler_step = lr_scheduler_step
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return torch.nn.functional.log_softmax(x, dim=1)

    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        dynamic = dynamic.float()
        static = static.float()
        y = y.long()
        logits = self.forward(torch.cat([dynamic, static], dim=1))
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        from sklearn.metrics import roc_auc_score
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        auc = self.val_auc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        auc = self.test_auc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_step,
                                                       gamma=self.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
