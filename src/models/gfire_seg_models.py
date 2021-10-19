from typing import Any, List
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUC, ConfusionMatrix, AUROC, AveragePrecision
from src.models.modules.fire_modules import SK_CNN, SK_LSTM, SK_CLSTM, SimpleLSTM, SimpleLSTMAttention, SimpleConvLSTM, \
    SimpleCNN, Resnet18CNN, \
    DynUnet, SimpleFCN
import torch
import numpy as np
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class UnetCNN_fire_model(LightningModule):
    """
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
            positive_weight: float = 0.5,
            hidden_size: int = 16,
            lstm_layers: int = 2,
            attention: bool = False,
            lr: float = 0.001,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # self.model = SimpleCNN(hparams=self.hparams)
        self.model = DynUnet(hparams=self.hparams)

        assert (positive_weight < 1) and (positive_weight > 0)
        self.positive_weight = positive_weight

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        # self.criterion = torch.nn.NLLLoss()
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        # self.train_auc = AUROC(pos_label=1)
        # self.train_auprc = AveragePrecision()

        self.val_accuracy = Accuracy()
        # self.val_auc = AUROC(pos_label=1)
        # self.val_auprc = AveragePrecision()

        self.test_accuracy = Accuracy()
        # self.test_auc = AUROC(pos_label=1)
        # self.test_auprc = AveragePrecision()

    def _log_step(self, phase, loss, preds, preds_proba, targets):
        for metric_name, metric_fn in self.logit_metrics_dict[phase].items():
            value = metric_fn(preds, targets)
            self.log(f'{phase}/{metric_name}', value, on_step=False, on_epoch=True, prog_bar=True)

        for metric_name, metric_fn in self.proba_metrics_dict[phase].items():
            value = metric_fn(preds_proba, targets)
            self.log(f'{phase}/{metric_name}', value, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        dynamic = dynamic.float()
        static = static.float()
        y = y.long()
        logits = self.forward(torch.cat([dynamic, static], dim=1))
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        loss = self.criterion(logits, y)
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'train'


        # log train metrics
        acc = self.train_accuracy(preds, targets)
        # auc = self.train_auc(preds_proba, targets)
        # auprc = self.train_auprc(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        # auc = self.train_auc(preds_proba, targets)
        # auprc = self.train_auprc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        # auc = self.train_auc(preds_proba, targets)
        # auprc = self.train_auprc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step,
                                                       gamma=self.hparams.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
