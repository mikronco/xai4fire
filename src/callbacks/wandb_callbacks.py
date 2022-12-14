import glob
import os
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import gc
from src.datamodules.greecefire_datamodule import FireDatasetWholeDay


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


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(targets, preds, average=None)
            r = recall_score(targets, preds, average=None)
            p = precision_score(targets, preds, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, axis=-1)

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )


class LogValPredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.preds_proba = []
        self.targets = []

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds_proba.append(outputs["preds_proba"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            preds_proba = torch.cat(self.preds_proba[:self.num_samples]).cpu().numpy()
            targets = torch.cat(self.targets[:self.num_samples]).cpu().numpy()

            imgs = []
            for i in range(self.num_samples):
                imgs.append(wandb.Image(plt.imshow(targets[i].squeeze(), cmap='Spectral_r'), caption=f'Target {i}'))
                imgs.append(
                    wandb.Image(plt.imshow(preds_proba[i].squeeze(), cmap='Spectral_r'), caption=f'Prediction {i}'))

            # log the images as wandb Image
            experiment.log(
                {
                    f"Danger Map": imgs
                }
            )
            self.preds_proba.clear()
            self.targets.clear()


class LogMapPredictions(Callback):
    """Logs a map prediction image to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, days: list, access_mode: str, dynamic_features: list, static_features: list, batch_size: int,
                 num_workers: int, nan_fill: float, problem_class: str):
        super().__init__()
        self.lag, self.patch_size = (0, 0)
        self.access_mode = access_mode
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nan_fill = nan_fill
        self.problem_class = problem_class
        if access_mode == 'temporal':
            self.lag, self.patch_size = (10, 0)
        if access_mode == 'spatial':
            self.lag, self.patch_size = (0, 25)
        if access_mode == 'spatiotemporal':
            self.lag, self.patch_size = (10, 25)
        self.days = days

        self.ready = True
        self.problem_class = problem_class
        self.override_whole = (problem_class == 'segmentation')

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        print("LogMapPredictionsCNN callback")
        if self.ready:
            device = pl_module.device
            # pl_module.to('cpu')
            if pl_module.on_gpu:
                torch.cuda.empty_cache()

            pl_module.eval()

            for day in self.days:
                day = int(day)
                print(day)
                # get a validation batch from the validation dat loader
                outputs = []
                dataset = FireDatasetWholeDay(day, self.access_mode, self.problem_class, self.patch_size, self.lag,
                                              self.dynamic_features,
                                              self.static_features,
                                              self.nan_fill)
                len_x = dataset.len_x
                len_y = dataset.len_y
                self.num_iterations = max(1, len(dataset) // self.batch_size)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=48,
                                        pin_memory=False)
                for i, (dynamic, static) in tqdm(enumerate(dataloader), total=self.num_iterations):
                    if self.override_whole:
                        dynamic = dynamic.float()
                        static = static.float()
                        if self.access_mode == 'spatial':
                            inputs = torch.cat([dynamic, static], dim=1)
                        else:
                            print("Shapes", dynamic.shape, static.shape)
                            bsize, timesteps, _, _, _ = dynamic.shape
                            static = static.unsqueeze(dim=1)
                            repeat_list = [1 for _ in range(static.dim())]
                            repeat_list[1] = timesteps
                            static = static.repeat(repeat_list)
                            inputs = torch.cat([dynamic, static], dim=2).float()
                        if pl_module.on_gpu:
                            inputs = inputs.cuda()
                        logits = pl_module(inputs).squeeze()
                        preds_proba = torch.exp(logits[1])
                        im = plt.imshow(preds_proba.detach().cpu().numpy().squeeze(), cmap='Spectral_r')
                        # Log the plot
                        experiment.log({f"Daily Danger Map {day}": im})
                        continue
                    if self.access_mode == 'spatial':
                        dynamic = dynamic.float()
                        static = static.float()
                        inputs = torch.cat([dynamic, static], dim=1)
                    if self.access_mode == 'temporal':
                        bsize, timesteps, _ = dynamic.shape
                        static = static.unsqueeze(dim=1)
                        repeat_list = [1 for _ in range(static.dim())]
                        repeat_list[1] = timesteps
                        static = static.repeat(repeat_list)
                        inputs = torch.cat([dynamic, static], dim=2).float()
                    if self.access_mode == 'spatiotemporal':
                        bsize, timesteps, _, _, _ = dynamic.shape
                        static = static.unsqueeze(dim=1)
                        repeat_list = [1 for _ in range(static.dim())]
                        repeat_list[1] = timesteps
                        static = static.repeat(repeat_list)
                        inputs = torch.cat([dynamic, static], dim=2).float()
                    if pl_module.on_gpu:
                        inputs = inputs.cuda()
                    logits = pl_module(inputs)
                    preds_proba = torch.exp(logits)[:, 1]
                    outputs.append(preds_proba.detach().cpu())
                if self.override_whole:
                    continue
                outputs = torch.cat(outputs, dim=0)
                outputs = torch.tensor(outputs)
                outputs = outputs.reshape(len_y, len_x)
                im = plt.imshow(outputs.detach().cpu().numpy().squeeze(), cmap='Spectral_r')
                plt.axis('off')
                plt.tight_layout()
                # Log the plot
                experiment.log({f"Fire Danger Map {day}": im})
                plt.clf()
            pl_module.to(device)


from src.utils.plotting import lime_feature_ranking


class LogLimeFR(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples=8):
        super().__init__()
        self.ready = True
        self.num_samples = num_samples

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_dynamic, val_static, _, val_labels = val_samples
            l = []
            sample_size = min(self.num_samples, len(val_labels))
            for i in range(sample_size):
                l.append(lime_feature_ranking(pl_module,
                                              val_dynamic[i].unsqueeze(0).to(device=pl_module.device),
                                              val_static[i].unsqueeze(0).to(device=pl_module.device),
                                              val_labels[i].to(device=pl_module.device),
                                              pl_module.hparams['dynamic_features'] + pl_module.hparams[
                                                  'static_features'],
                                              pl_module.hparams['access_mode'])
                         )

            # # run the batch through the network
            # val_imgs = val_imgs.to(device=pl_module.device)
            # logits = pl_module(val_imgs)
            # preds = torch.argmax(logits, axis=-1)

            # log the images as wandb Image
            # experiment.log({'Lime Feature Ranking': wandb.Image(fig)})
            # log the images as wandb Image
            experiment.log({f"Images/{experiment.name}":
                                [wandb.Image(x, caption=f"Lime - Pred:{pred:.2f} / Label:{y}") for x, pred, y in l]})
            plt.clf()
