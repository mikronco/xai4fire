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
from torchvision.models.resnet import resnet18
from fastai.vision.models import unet


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


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

        # m = resnet18(pretrained=False)
        # m.conv1 = nn.Conv2d(hidden_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # num_ftrs = m.fc.in_features
        # m.fc = nn.Linear(num_ftrs, 2)
        # self.m = m
        # cnn part
        # self.conv1 = nn.Conv2d(hidden_size, 8, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # fully-connected part

        kernel_size = 3
        # self.se1 = SE_Block(hidden_size, r=4)
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=1)
        # self.se2 = SE_Block(hidden_size, r=4)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(kernel_size * kernel_size * hidden_size * 16, 16)
        # self.fc1 = nn.Linear(10000, 16)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 8)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor):
        _, last_states = self.convlstm(x)
        x = last_states[0][0]
        # x = self.se1(x)
        x = F.relu(self.conv1(x))
        # x = self.se2(x)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop1(self.fc2(x)))
        x = self.fc3(x)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # x = torch.flatten(x, 1)
        # x = F.relu(self.drop1(self.fc1(x)))
        # x = F.relu(self.drop2(self.fc2(x)))
        return torch.nn.functional.log_softmax(x, dim=1)


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


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths=None):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        if lengths:
            for i, l in enumerate(lengths):  # skip the first sentence
                if l < max_len:
                    mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class SimpleLSTMAttention(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        hidden_dim = hidden_size
        lstm_layers = hparams['lstm_layers']
        lstm_layer = 2
        self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_size,
                             num_layers=lstm_layers,
                             bidirectional=True)
        self.atten1 = Attention(hidden_dim * 2, batch_first=True)  # 2 is bidrectional
        self.lstm2 = nn.LSTM(input_size=hidden_dim * 2,
                             hidden_size=hidden_dim,
                             num_layers=1,
                             bidirectional=True)
        self.atten2 = Attention(hidden_dim * 2, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim * lstm_layer * 2, hidden_dim * lstm_layer * 2),
                                 nn.BatchNorm1d(hidden_dim * lstm_layer * 2),
                                 nn.ReLU())
        self.fc2 = nn.Linear(hidden_dim * lstm_layer * 2, 2)

    def forward(self, x):
        out1, (h_n, c_n) = self.lstm1(x)
        x, _ = self.atten1(out1)  # skip connect

        out2, (h_n, c_n) = self.lstm2(out1)
        y, _ = self.atten2(out2)

        z = torch.cat([x, y], dim=1)
        z = self.fc1(self.dropout(z))
        z = self.fc2(self.dropout(z))
        return torch.nn.functional.log_softmax(z, dim=1)


class SK_CLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.convlstm = ConvLSTM(input_dim, hidden_size, (3, 3), 1, True, True, False)
        self.fc1 = nn.Linear(hidden_size * 25 * 25, 16)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 8)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        _, last_states = self.convlstm(x)
        x = F.relu(last_states[0][0])
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class SK_LSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.lstm = nn.LSTM(input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.drop1 = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(hidden_size // 2, 2)
        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.drop1,
            self.relu,
            self.fc2,
            self.drop2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)


class SK_CNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        kernel_size = 3
        self.conv1 = nn.Conv2d(input_dim, hidden_size, kernel_size=kernel_size, stride=1, padding=1)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(kernel_size * kernel_size * hidden_size * 16, 16)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 8)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop1(self.fc2(x)))
        x = self.fc3(x)
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


class SimpleFCN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        # CNN definition
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        self.conv1 = nn.Conv2d(input_dim, 8, kernel_size=(6, 6), stride=(1, 1), padding=(1, 1))
        self.conv11 = nn.Conv2d(8, 8, kernel_size=(7, 7), stride=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(9, 9), stride=(1, 1), padding=(1, 1))
        self.conv22 = nn.Conv2d(16, 16, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.conv33 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        k = 4
        self.fconv = nn.Conv2d(16, k * k * 16, kernel_size=(k, k), stride=(1, 1), padding=(0, 0))
        self.conv1_1x1 = nn.Conv2d(k * k * 16, 64, (1, 1))
        self.drop1 = nn.Dropout(0.5)
        self.conv2_1x1 = nn.Conv2d(64, 32, (1, 1))
        self.drop2 = nn.Dropout(0.5)
        self.conv3_1x1 = nn.Conv2d(32, 2, (1, 1))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv33(x))
        print(x.shape)
        x = self.fconv(x)
        print(x.shape)

        x = F.relu(self.drop1(self.conv1_1x1(x)))
        print(x.shape)

        x = F.relu(self.drop2(self.conv2_1x1(x)))
        print(x.shape)

        return torch.nn.functional.log_softmax(self.conv3_1x1(x), dim=1)


class DynUnet(nn.Module):
    def __init__(self, hparams: dict):
        super(DynUnet, self).__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        m = resnet18()
        m = nn.Sequential(*list(m.children())[:-2])
        m[0] = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = unet.DynamicUnet(m, 2, (25, 25), norm_type=None)

    def forward(self, x):
        out = self.net(x)
        return torch.nn.functional.log_softmax(out, dim=1)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class DynConvLSTMUnet(nn.Module):
    def __init__(self, hparams: dict):
        super(DynUnet, self).__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        m = resnet18()
        m = nn.Sequential(*list(m.children())[:-2])
        m[0] = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = unet.DynamicUnet(m, 2, (25, 25), norm_type=None)

        def forward(self, x):
            out = self.net(x)
            return torch.nn.functional.log_softmax(out, dim=1)

    # self.convlstm = ConvLSTM(input_dim,
    #                          hidden_size,
    #                          (3, 3),
    #                          lstm_layers,
    #                          True,
    #                          True,
    #                          False)
    #
    # _, last_states = self.convlstm(x)
    # x = last_states[0][0]


class Resnet18CNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # CNN definition
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        m = resnet18(pretrained=False)
        m.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, 2)
        self.m = m

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.log_softmax(self.m(x), dim=1)
