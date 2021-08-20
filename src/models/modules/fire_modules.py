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
from src.models.modules.convlstm import ConvLSTM

class SimpleConvLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        # clstm part
        self.convlstm = ConvLSTM(input_dim,
                                 hidden_size,
                                 (3, 3),
                                 lstm_layers,
                                 True,
                                 True,
                                 False)
        # cnn part
        self.conv1 = nn.Conv2d(hidden_size, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # fully-connected part
        self.fc1 = nn.Linear(576, 64)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor):
        _, last_states = self.convlstm(x)
        x = last_states[0][0]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        return torch.nn.functional.log_softmax(self.fc3(x), dim=1)

class SimpleLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers=lstm_layers, batch_first=True)
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

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)


class SimpleCNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # CNN definition
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        self.conv1 = nn.Conv2d(input_dim, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(576, 64)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        return torch.nn.functional.log_softmax(self.fc3(x), dim=1)
