from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datamodules.datasets.greecefire_dataset import FireDatasetWholeDay, FireDS


class FireDSDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            access_mode: str = 'temporal',
            problem_class: str = 'classification',
            nan_fill: float = 1.,
            sel_dynamic_features=None,
            sel_static_features=None
    ):
        super().__init__()
        if sel_dynamic_features is None:
            sel_dynamic_features = []
        if sel_static_features is None:
            sel_static_features = []
        self.nan_fill = nan_fill
        self.access_mode = access_mode
        self.problem_class = problem_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.sel_dynamic_features = sel_dynamic_features
        self.sel_static_features = sel_static_features

        self.data_train = FireDS(access_mode=self.access_mode, problem_class=self.problem_class, train_val_test='train',
                                 dynamic_features=self.sel_dynamic_features,
                                 static_features=self.sel_static_features,
                                 categorical_features=None, nan_fill=self.nan_fill)
        self.data_val = FireDS(access_mode=self.access_mode, problem_class=self.problem_class, train_val_test='val',
                               dynamic_features=self.sel_dynamic_features,
                               static_features=self.sel_static_features,
                               categorical_features=None, nan_fill=self.nan_fill)
        self.data_test = FireDS(access_mode=self.access_mode, problem_class=self.problem_class, train_val_test='test',
                                dynamic_features=self.sel_dynamic_features,
                                static_features=self.sel_static_features,
                                categorical_features=None, nan_fill=self.nan_fill)

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )