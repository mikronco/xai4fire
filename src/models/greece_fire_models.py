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


class ConvLSTM_fire_model(LightningModule):
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
            hidden_size: int = 32,
            lstm_layers: int = 1,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SimpleConvLSTM(hparams=self.hparams)
        # self.model = SK_CLSTM(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1. - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        self.train_auc = AUROC(pos_label=1)
        self.train_auprc = AveragePrecision()

        self.val_accuracy = Accuracy()
        self.val_auc = AUROC(pos_label=1)
        self.val_auprc = AveragePrecision()

        self.test_accuracy = Accuracy()
        self.test_auc = AUROC(pos_label=1)
        self.test_auprc = AveragePrecision()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        dynamic, static, _, y = batch
        y = y.long()
        bsize, timesteps, _, _, _ = dynamic.shape
        static = static.unsqueeze(dim=1)
        repeat_list = [1 for _ in range(static.dim())]
        repeat_list[1] = timesteps
        static = static.repeat(repeat_list)
        inputs = torch.cat([dynamic, static], dim=2).float()
        logits = self.forward(inputs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'train'


        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step,
                                                       gamma=self.hparams.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class LSTM_fire_model(LightningModule):
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
            hidden_size: int = 32,
            lstm_layers: int = 3,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            attention: bool = False
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.attention = attention
        if self.attention:
            self.model = SimpleLSTMAttention(hparams=self.hparams)
        else:
            self.model = SK_LSTM(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        self.train_auc = AUROC(pos_label=1)
        self.train_auprc = AveragePrecision()

        self.val_accuracy = Accuracy()
        self.val_auc = AUROC(pos_label=1)
        self.val_auprc = AveragePrecision()

        self.test_accuracy = Accuracy()
        self.test_auc = AUROC(pos_label=1)
        self.test_auprc = AveragePrecision()

    def forward(self, x: torch.Tensor):
        return self.model(x)

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
        phase = 'train'


        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
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
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        self.train_auc = AUROC(pos_label=1)
        self.train_auprc = AveragePrecision()

        self.val_accuracy = Accuracy()
        self.val_auc = AUROC(pos_label=1)
        self.val_auprc = AveragePrecision()

        self.test_accuracy = Accuracy()
        self.test_auc = AUROC(pos_label=1)
        self.test_auprc = AveragePrecision()

    def forward(self, x: torch.Tensor):
        return self.model(x)

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
        phase = 'train'


        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
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


class CNN_fire_model(LightningModule):
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
            positive_weight: float = 0.8,
            hidden_size: int = 32,
            lr: float = 0.001,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SK_CNN(hparams=self.hparams)
        # self.model = Resnet18CNN(hparams=self.hparams)

        assert (positive_weight < 1) and (positive_weight > 0)
        self.positive_weight = positive_weight

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        self.train_auc = AUROC(pos_label=1)
        self.train_auprc = AveragePrecision()

        self.val_accuracy = Accuracy()
        self.val_auc = AUROC(pos_label=1)
        self.val_auprc = AveragePrecision()

        self.test_accuracy = Accuracy()
        self.test_auc = AUROC(pos_label=1)
        self.test_auprc = AveragePrecision()

    def forward(self, x: torch.Tensor):
        return self.model(x)

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
        phase = 'train'


        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
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


class FCN_fire_model(LightningModule):
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
            positive_weight: float = 0.8,
            lr: float = 0.001,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # self.model = SimpleCNN(hparams=self.hparams)
        self.model = SimpleFCN(hparams=self.hparams)

        assert (positive_weight < 1) and (positive_weight > 0)
        self.positive_weight = positive_weight

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy()
        self.train_auc = AUROC(pos_label=1)
        self.train_auprc = AveragePrecision()

        self.val_accuracy = Accuracy()
        self.val_auc = AUROC(pos_label=1)
        self.val_auprc = AveragePrecision()

        self.test_accuracy = Accuracy()
        self.test_auc = AUROC(pos_label=1)
        self.test_auprc = AveragePrecision()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        dynamic = dynamic.float()
        static = static.float()
        y = y.long()
        logits = self.forward(torch.cat([dynamic, static], dim=1)).squeeze()
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'train'


        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        auc = self.train_auc(preds_proba, targets)
        auprc = self.train_auprc(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)
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
