import torch
import csv
import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import os
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path

dataset_path = Path.home() / 'jh-shared/iprapas/uc3'
root = dataset_path / 'datasets'

all_dynamic_features = [
    'Fpar_500m',
    'LST_Day_1km',
    'LST_Night_1km',
    '1 km 16 days NDVI',
    'ET_500m',
    'era5_max_u10',
    'era5_max_v10',
    'era5_max_t2m',
    'era5_max_tp',
    'era5_min_u10',
    'era5_min_v10',
    'era5_min_t2m',
    'era5_min_tp',
    #     'fwi',
    #     'danger_risk'
]

coordinates = ['x', 'y']

all_static_features = ['dem_mean',
                       'dem_std',
                       'aspect_mean',
                       'aspect_std',
                       'slope_mean',
                       'slope_std',
                       'roads_density_2020',
                       'population_density']

target = 'burned_areas'

ds = xr.open_dataset(dataset_path / 'dataset_greece_unzipped.nc')


class FireDS(Dataset):
    def __init__(self, data_dir: Path = root, access_mode: str = 'spatiotemporal',
                 problem_class: str = 'classification',
                 train: bool = True, dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1.):
        """
        @param data_dir: root path that containts dataset as netcdf (.nc) files
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train: True gets samples from [2009-2019]. False get samples from 2020
        @param dynamic_features: selects the dynamic features to return
        @param static_features: selects the static features to return
        @param categorical_features: selects the categorical features
        @param nan_fill: Fills nan with the value specified here
        """
        if static_features is None:
            static_features = all_static_features
        if dynamic_features is None:
            dynamic_features = all_dynamic_features
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.categorical_features = categorical_features
        self.access_mode = access_mode
        self.problem_class = problem_class
        self.data_dir = data_dir
        self.nan_fill = nan_fill
        assert problem_class in ['classification', 'segmentation']
        if problem_class == 'classification':
            self.target = 'burned'
        else:
            self.target = 'burned_areas'
        assert self.access_mode in ['spatial', 'temporal', 'spatiotemporal']
        if access_mode == 'spatial':
            ds_path = data_dir / 'spatial_dataset_clc_25x25_wmean.nc'
        elif access_mode == 'temporal':
            ds_path = data_dir / 'temporal_dataset_clc_10_wmean.nc'
        else:
            ds_path = data_dir / 'spatiotemporal_dataset_clc_10x25x25_wmean.nc'

        self.ds_orig = ds
        self.ds = xr.open_dataset(ds_path)

        # TODO change time split to account for date from time_
        self.train_ds = self.ds.isel(patch=slice(0, 17068))
        self.test_ds = self.ds.isel(patch=slice(17068, len(self.ds.patch)))

        if train:
            self.ds = self.train_ds
        else:
            self.ds = self.test_ds

        self.ds = self.ds.load()

    def __len__(self):
        return len(self.ds.patch)

    def __getitem__(self, idx):
        chunk = self.ds.isel(patch=idx)

        dynamic = np.stack([norm_ds(chunk, chunk, feature) for feature in self.dynamic_features])
        # lstm, convlstm expect input that has the time dimension first
        if 'temp' in self.access_mode:
            dynamic = np.moveaxis(dynamic, 0, 1)
        static = np.stack([norm_ds(chunk, chunk, feature) for feature in self.static_features])
        labels = chunk[self.target].values
        if self.nan_fill:
            dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
            static = np.nan_to_num(static, nan=self.nan_fill)
        return dynamic, static, 0, labels


def norm_ds(input_dataset: xr.Dataset, mean_dataset: xr.Dataset, feature: str):
    return (input_dataset[feature] - mean_dataset[feature + '_mean']) / mean_dataset[feature + '_std']


def get_pixel_feature_ds(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0):
    assert access_mode in ['spatial', 'temporal', 'spatiotemporal']
    assert lag >= 0 and patch_size >= 0 and t >= 0 and x >= 0 and y >= 0
    patch_half = patch_size // 2
    assert x >= patch_half and x + patch_half < the_ds.dims['x']
    assert y >= patch_half and y + patch_half < the_ds.dims['y']
    #     len_x = ds.dims['x'] - patch_size
    #     len_y = ds.dims['y'] - patch_size
    if access_mode == 'spatiotemporal':
        block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=slice(x - patch_half, x + patch_half + 1),
                            y=slice(y - patch_half, y + patch_half + 1)).reset_index(['x', 'y', 'time'])
    elif access_mode == 'temporal':
        block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=x, y=y).reset_index(['time'])
    elif access_mode == 'spatial':
        block = the_ds.isel(x=slice(x - patch_half, x + patch_half + 1),
                            y=slice(y - patch_half, y + patch_half + 1)).reset_index(['x', 'y'])
    if access_mode == 'spatial':
        year = pd.DatetimeIndex([the_ds['time'].values]).year[0]
    else:
        year = pd.DatetimeIndex([the_ds['time'][t].values]).year[0]

    # clc
    if year < 2012:
        block['clc'] = block['clc_2006']
    elif year < 2018:
        block['clc'] = block['clc_2012']
    else:
        block['clc'] = block['clc_2018']

    # population density
    block['population_density'] = block[f'population_density_{year}']
    return block


def get_pixel_feature_vector(the_ds, mean_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0,
                             dynamic_features=None,
                             static_features=None, nan_fill=-1.0, override_whole=False):
    if override_whole:
        chunk = the_ds
    else:
        chunk = get_pixel_feature_ds(the_ds, t=t, x=x, y=y, access_mode=access_mode, patch_size=patch_size, lag=lag)
    dynamic = np.stack([norm_ds(chunk, mean_ds, feature) for feature in dynamic_features])
    if 'temp' in access_mode:
        dynamic = np.moveaxis(dynamic, 0, 1)
    static = np.stack([norm_ds(chunk, mean_ds, feature) for feature in static_features])
    if nan_fill:
        dynamic = np.nan_to_num(dynamic, nan=nan_fill)
        static = np.nan_to_num(static, nan=nan_fill)
    return dynamic, static


# class FireDatasetOneDay(Dataset):
#     def __init__(self, days, access_mode='temporal', patch_size=0, lag=10, dynamic_features=None,
#                  static_features=None, nan_fill=-1.0):
#
#         self.ds = ds.isel(time=days)
#         self.target = target
#         self.dynamic_features = dynamic_features
#         self.static_features = static_features
#         dataset_path = Path.home() / f'jh-shared/iprapas/uc3/datasets/spatial_dataset_clc_25x25_wmean.nc'
#         self.mean_ds = xr.open_dataset(dataset_path).load()
#         self.x_dim = ds.dims['x']
#         self.y_dim = ds.dims['y']
#
#     def __len__(self):
#         return len(self.ds.time)
#
#     def __getitem__(self, idx):
#         chunk = self.ds.isel(time=idx)
#
#         dynamic = np.stack([norm_ds(chunk, self.mean_ds, feature) for feature in self.dynamic_features])
#         static = np.stack([norm_ds(chunk, self.mean_ds, feature) for feature in self.static_features])
#
#
#         target = chunk[self.target].values
#         target[np.isnan(target)] = 0
#
#         return chunk['time'].values, dynamic, static, target


class FireDatasetWholeDay(Dataset):
    def __init__(self, day, access_mode='temporal', patch_size=0, lag=10, dynamic_features=None,
                 static_features=None, nan_fill=-1.0, override_whole=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            dynamic_transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if lag > 0:
            self.ds = ds.isel(time=range(day - lag + 1, day + 1)).load()
        else:
            self.ds = ds.isel(time=day).load()
        s = ''
        if lag:
            s = f'{lag}'
            if patch_size:
                s = f'{lag}x{patch_size}x{patch_size}'
        elif patch_size:
            s = f'{patch_size}x{patch_size}'
        self.override_whole = override_whole
        dataset_path = Path.home() / f'jh-shared/iprapas/uc3/datasets/{access_mode}_dataset_{s}_wmean.nc'
        self.mean_ds = xr.open_dataset(dataset_path).load()
        self.ds = self.ds.load()
        self.len_x = ds.dims['x'] - patch_size
        self.len_y = ds.dims['y'] - patch_size
        self.patch_size = patch_size
        self.lag = lag
        self.access_mode = access_mode
        self.day = day
        self.nan_fill = nan_fill
        self.dynamic_features = dynamic_features
        self.static_features = static_features

    def __len__(self):
        if self.override_whole:
            return 1
        return self.len_x * self.len_y

    def __getitem__(self, idx):
        y = idx // self.len_x
        x = idx % self.len_x
        if self.lag == 0:
            day = 0
        else:
            day = self.lag - 1
        dynamic, static = get_pixel_feature_vector(self.ds, self.mean_ds, day, x + self.patch_size // 2,
                                                   y + self.patch_size // 2, self.access_mode, self.patch_size,
                                                   self.lag,
                                                   self.dynamic_features,
                                                   self.static_features,
                                                   self.nan_fill, self.override_whole)
        return dynamic, static
