import torch
import csv
import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import fsspec
import zarr
import os
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import random
import json
import random

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

run = 'noa'
datacube_path = Path.home() / 'jh-shared/iprapas/uc3'
if run == 'remote':
    url = 'https://storage.de.cloud.ovh.net/v1/AUTH_84d6da8e37fe4bb5aea18902da8c1170/uc3/uc3cube.zarr'
    ds = xr.open_zarr(fsspec.get_mapper(url), consolidated=True)
else:
    ds = xr.open_dataset(datacube_path / 'greece_big_zipped.nc')

dataset_root = datacube_path / 'datasets'
ds_paths = \
    {
        'classification':
            {'spatial': dataset_root / 'new_spatial_dataset_clc_25x25_wmean.nc',
             'temporal': dataset_root / 'new_temporal_dataset_clc_10_wmean.nc',
             'spatiotemporal': dataset_root / 'spatiotemporal_dataset_clc_10x25x25_wmean.nc'},
        'segmentation':
            {'spatial': dataset_root / 'new_spatial_dataset_clc_25x25_wmean.nc',
             'spatiotemporal': dataset_root / '../chunked_new_segmentation_spatiotemporal_dataset_clc_10x125x125_wmean.nc'}
    }


def norm_ds(input_dataset: xr.Dataset, mean_dataset: xr.Dataset, feature: str):
    return (input_dataset[feature] - mean_dataset[feature + '_mean']) / mean_dataset[feature + '_std']


class FireDSnp(Dataset):
    def __init__(self, access_mode: str = 'spatiotemporal',
                 problem_class: str = 'classification',
                 train_val_test: str = 'train', dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1.):
        """
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2018].
                'val' gets samples from 2019.
                test' get samples from 2020
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
        self.nan_fill = nan_fill
        assert problem_class in ['classification', 'segmentation']
        if problem_class == 'classification':
            self.target = 'burned'
        else:
            self.target = 'burned_areas'
        assert self.access_mode in ['spatial', 'temporal', 'spatiotemporal']
        dataset_path = dataset_root / self.access_mode
        self.path_list = [x for x in (dataset_root / self.access_mode).glob('*')]
        self.train_path_list = [x for x in dataset_path.glob('*') if x.stem[:4] not in ['2020', '2021', '2019']]
        self.val_path_list = [x for x in dataset_path.glob('2019*')]
        self.test_path_list = [x for x in dataset_path.glob('2020*')]

        if train_val_test == 'train':
            self.path_list = self.train_path_list
        elif train_val_test == 'val':
            self.path_list = self.val_path_list
        elif train_val_test == 'test':
            self.path_list = self.test_path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        chunk = np.load(self.path_list[idx])

        dynamic = np.stack([chunk[feature] for feature in self.dynamic_features])
        # lstm, convlstm expect input that has the time dimension first
        if 'temp' in self.access_mode:
            dynamic = np.moveaxis(dynamic, 0, 1)
        static = np.stack([chunk[feature] for feature in self.static_features])
        labels = chunk[self.target]
        if self.nan_fill:
            dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
            static = np.nan_to_num(static, nan=self.nan_fill)
        labels = np.nan_to_num(labels, nan=0.0)
        return dynamic, static, 0, labels


class FireDS(Dataset):
    def __init__(self, access_mode: str = 'spatiotemporal',
                 problem_class: str = 'classification',
                 train_val_test: str = 'train', dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1.):
        """
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2018].
                'val' gets samples from 2019.
                test' get samples from 2020
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
        self.nan_fill = nan_fill
        assert problem_class in ['classification', 'segmentation']
        if problem_class == 'classification':
            self.target = 'burned'
        else:
            self.target = 'burned_areas'
        assert self.access_mode in ['spatial', 'temporal', 'spatiotemporal']

        ds_path = ds_paths[problem_class][access_mode]

        self.ds_orig = ds
        self.ds = xr.open_dataset(ds_path)
        if access_mode == 'spatial':
            dates = pd.DatetimeIndex(self.ds.time.values)
        else:
            dates = pd.DatetimeIndex(self.ds.isel(time=0).time_.values)

        val_year = 2019
        test_year = 2020
        self.train_ds = self.ds.isel(patch=list(np.where(~dates.year.isin([val_year, test_year, 2021]))[0]))
        self.val_ds = self.ds.isel(patch=list(np.where(dates.year == val_year)[0]))
        self.test_ds = self.ds.isel(patch=list(np.where(dates.year == test_year)[0]))

        if train_val_test == 'train':
            self.ds = self.train_ds
        elif train_val_test == 'val':
            self.ds = self.val_ds
        elif train_val_test == 'test':
            self.ds = self.test_ds

        load = False
        print("Loading dataset")
        print(self.ds)
        if load:
            self.ds = self.ds.load()
        print("Dataset loaded")

    def __len__(self):
        return len(self.ds.patch)

    def __getitem__(self, idx):

        chunk = self.ds.isel(patch=idx).load()
        dynamic = np.stack([norm_ds(chunk, chunk, feature) for feature in self.dynamic_features])
        # lstm, convlstm expect input that has the time dimension first
        if 'temp' in self.access_mode:
            dynamic = np.moveaxis(dynamic, 0, 1)
        static = np.stack([norm_ds(chunk, chunk, feature) for feature in self.static_features])
        labels = chunk[self.target].values
        # if self.nan_fill:
        dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
        static = np.nan_to_num(static, nan=self.nan_fill)

        print(dynamic[0])
        if self.problem_class == 'segmentation':
            labels[labels == 2] = 1
            labels = np.nan_to_num(labels, nan=0.0)
            crop_size = 65
            random_start_x = random.randint(0, 125 - crop_size)
            random_start_y = random.randint(0, 125 - crop_size)
            dynamic = dynamic[:, :, random_start_x:random_start_x + crop_size,
                      random_start_y:random_start_y + crop_size]
            static = static[:, random_start_x:random_start_x + crop_size, random_start_y:random_start_y + crop_size]
            labels = labels[random_start_x:random_start_x + crop_size, random_start_y:random_start_y + crop_size]
        return dynamic, static, 0, labels


dataset_root = Path.home() / 'hdd1/iprapas/uc3/datasets_v2'
min_max_file = dataset_root / 'minmax_clc.json'
variable_file = dataset_root / 'variable_dict.json'
with open(min_max_file) as f:
    min_max_dict = json.load(f)

with open(variable_file) as f:
    variable_dict = json.load(f)


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
                            y=slice(y - patch_half, y + patch_half + 1))  # .reset_index(['x', 'y', 'time'])
    elif access_mode == 'temporal':
        block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=x, y=y).reset_index(['time'])
    elif access_mode == 'spatial':
        block = the_ds.isel(x=slice(x - patch_half, x + patch_half + 1),
                            y=slice(y - patch_half, y + patch_half + 1))  # .reset_index(['x', 'y'])

    return block


def get_pixel_feature_vector(the_ds, mean_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0,
                             dynamic_features=None,
                             static_features=None, nan_fill=-1.0, override_whole=False, scaling='minmax'):
    if override_whole:
        chunk = the_ds
    else:
        chunk = get_pixel_feature_ds(the_ds, t=t, x=x, y=y, access_mode=access_mode, patch_size=patch_size, lag=lag)

    if scaling == 'norm':
        dynamic = np.stack([norm_ds(chunk, mean_ds, feature) for feature in dynamic_features])
        static = np.stack([norm_ds(chunk, mean_ds, feature) for feature in static_features])
    elif scaling == 'minmax':
        dynamic = np.stack([min_max_scaling(chunk, feature, access_mode) for feature in dynamic_features])
        static = np.stack([min_max_scaling(chunk, feature, access_mode) for feature in static_features])

    if 'temp' in access_mode:
        dynamic = np.moveaxis(dynamic, 0, 1)

    dynamic = np.nan_to_num(dynamic, nan=nan_fill)
    static = np.nan_to_num(static, nan=nan_fill)
    return dynamic, static


class FireDatasetWholeDay(Dataset):
    def __init__(self, day, access_mode='temporal', problem_class='classification', patch_size=0, lag=10,
                 dynamic_features=None,
                 static_features=None, nan_fill=-1.0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            dynamic_transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert access_mode in ['temporal', 'spatial', 'spatiotemporal']
        assert problem_class in ['classification', 'segmentation']
        self.problem_class = problem_class
        if lag > 0:
            self.ds = ds.isel(time=range(day - lag + 1, day + 1))
        else:
            self.ds = ds.isel(time=day)
        self.override_whole = problem_class == 'segmentation'
        mean_ds_path = ds_paths[problem_class][access_mode]

        self.mean_ds = xr.open_dataset(mean_ds_path)
        print(self.ds)
        self.ds = self.ds.load()
        print("Dataset loaded...")
        pixel_range = patch_size // 2
        self.pixel_range = pixel_range
        self.len_x = self.ds.dims['x']
        self.len_y = self.ds.dims['y']
        if access_mode == 'spatial':
            year = pd.DatetimeIndex([self.ds['time'].values]).year[0]
        else:
            year = pd.DatetimeIndex([self.ds['time'][0].values]).year[0]
        # clc
        if 'clc' in static_features:
            if year < 2012:
                self.ds['clc'] = self.ds['clc_2006']
            elif year < 2018:
                self.ds['clc'] = self.ds['clc_2012']
            else:
                self.ds['clc'] = self.ds['clc_2018']

        # population density
        if 'population_density' in static_features:
            self.ds['population_density'] = self.ds[f'population_density_{year}']

        if access_mode == 'spatiotemporal':
            new_ds_dims = ['time', 'y', 'x']
            new_ds_dict = {}
            print("Padding dynamic features...")

            for feat in dynamic_features:
                new_ds_dict[feat] = (new_ds_dims,
                                     np.pad(self.ds[feat].values,
                                            pad_width=((0, 0), (pixel_range, pixel_range), (pixel_range, pixel_range)),
                                            mode='constant', constant_values=(0, 0)))
            new_ds_dims_static = ['y', 'x']
            print("Padding static features...")

            for feat in static_features:
                new_ds_dict[feat] = (new_ds_dims_static,
                                     np.pad(self.ds[feat].values,
                                            pad_width=((pixel_range, pixel_range), (pixel_range, pixel_range)),
                                            mode='constant', constant_values=(0, 0)))
            self.ds = xr.Dataset(new_ds_dict)

        elif access_mode == 'spatial':
            new_ds_dims = ['y', 'x']
            new_ds_dict = {}
            for feat in dynamic_features + static_features:
                new_ds_dict[feat] = (new_ds_dims,
                                     np.pad(self.ds[feat].values,
                                            pad_width=((pixel_range, pixel_range), (pixel_range, pixel_range)),
                                            mode='constant', constant_values=(0, 0)))
            new_ds_dict['time'] = self.ds['time'].values
            self.ds = xr.Dataset(new_ds_dict)

        self.patch_size = patch_size
        self.lag = lag
        self.access_mode = access_mode
        self.day = day
        self.nan_fill = nan_fill
        self.dynamic_features = dynamic_features
        self.static_features = static_features

    def __len__(self):
        if self.problem_class == 'segmentation':
            return 1
        return self.len_x * self.len_y

    def __getitem__(self, idx):
        y = idx // self.len_x + self.pixel_range
        x = idx % self.len_x + self.pixel_range

        if self.lag == 0:
            day = 0
        else:
            day = self.lag - 1
        dynamic, static = get_pixel_feature_vector(self.ds, self.mean_ds, day, x,
                                                   y, self.access_mode, self.patch_size,
                                                   self.lag,
                                                   self.dynamic_features,
                                                   self.static_features,
                                                   self.nan_fill, self.override_whole)
        return dynamic, static


def min_max_scaling(chunk, feat_name, access_mode):
    '''
    (x - min)/(max - min)
    '''
    return (chunk[feat_name] - min_max_dict['min'][access_mode][feat_name]) / (
            min_max_dict['max'][access_mode][feat_name] - min_max_dict['min'][access_mode][feat_name])


class FireDataset_nc(Dataset):
    def __init__(self, access_mode: str = 'spatiotemporal',
                 problem_class: str = 'classification',
                 train_val_test: str = 'train', dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1.):
        """
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2018].
                'val' gets samples from 2019.
                test' get samples from 2020
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
        self.nan_fill = nan_fill
        assert problem_class in ['classification', 'segmentation']
        if problem_class == 'classification':
            self.target = 'burned'
        else:
            self.target = 'burned_areas'
        assert self.access_mode in ['spatial', 'temporal', 'spatiotemporal']
        dataset_path = dataset_root / self.access_mode
        self.positives_list = list((dataset_path / 'positives').glob('*.nc'))
        self.negatives_list = list((dataset_path / 'negatives_clc').glob('*.nc'))
        self.path_list = self.positives_list + self.negatives_list
        # self.mean_ds = xr.open_dataset(mean_ds_paths[problem_class][access_mode])
        self.train_path_list = [x for x in self.path_list if x.stem[:4] not in ['2019', '2020', '2021']]
        self.val_path_list = [x for x in self.path_list if x.stem[:4] == '2019']
        self.test_path_list = [x for x in self.path_list if x.stem[:4] == '2020']

        #         self.path_list = [x for x in (dataset_root / self.access_mode).glob('*')]
        #         self.train_path_list = [x for x in dataset_path.glob('*') if x.stem[:4] not in ['2020', '2021', '2019']]
        #         self.val_path_list = [x for x in dataset_path.glob('2019*')]
        #         self.test_path_list = [x for x in dataset_path.glob('2020*')]

        if train_val_test == 'train':
            self.path_list = self.train_path_list
        elif train_val_test == 'val':
            self.path_list = self.val_path_list
        elif train_val_test == 'test':
            self.path_list = self.test_path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        chunk = xr.open_dataset(self.path_list[idx]).load()

        dynamic = np.stack(
            [min_max_scaling(chunk, feature, self.access_mode) for feature in self.dynamic_features])
        #         dynamic = np.stack([chunk[feature] for feature in self.dynamic_features])
        #
        # lstm, convlstm expect input that has the time dimension first
        if 'temp' in self.access_mode:
            dynamic = np.moveaxis(dynamic, 0, 1)
        static = np.stack(
            [min_max_scaling(chunk, feature, self.access_mode) for feature in self.static_features])
        labels = chunk[self.target].values
        if self.nan_fill:
            dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
            static = np.nan_to_num(static, nan=self.nan_fill)
        labels = np.nan_to_num(labels, nan=0.)
        return dynamic, static, 0, labels


dataset_np_root = Path.home() / 'hdd1/iprapas/uc3/datasets_np'


class FireDataset_np(Dataset):
    def __init__(self, access_mode: str = 'spatiotemporal',
                 problem_class: str = 'classification',
                 train_val_test: str = 'train', dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1., neg2pos_ratio: int = 2):
        """
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2018].
                'val' gets samples from 2019.
                test' get samples from 2020
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
        self.nan_fill = nan_fill
        assert problem_class in ['classification', 'segmentation']
        if problem_class == 'classification':
            self.target = 'burned'
        else:
            self.target = 'burned_areas'
        assert self.access_mode in ['spatial', 'temporal', 'spatiotemporal']
        dataset_path = dataset_np_root / self.access_mode
        print(dataset_path)
        self.positives_list = list(dataset_path.glob('*1.npy'))

        all_negatives_list = list(dataset_path.glob('*0.npy'))
        num_positives = len(self.positives_list)
        num_negatives = len(all_negatives_list)
        neg_sample_size = min(num_negatives, neg2pos_ratio * num_positives)
        self.negatives_list = random.sample(all_negatives_list, neg_sample_size)
        self.path_list = self.positives_list + self.negatives_list

        self.train_path_list = [x for x in self.path_list if x.stem[:4] not in ['2019', '2020', '2021']]
        self.val_path_list = [x for x in self.path_list if x.stem[:4] == '2019']
        self.test_path_list = [x for x in self.path_list if x.stem[:4] == '2020']

        if train_val_test == 'train':
            self.path_list = self.train_path_list
        elif train_val_test == 'val':
            self.path_list = self.val_path_list
        elif train_val_test == 'test':
            self.path_list = self.test_path_list
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        labels = int(self.path_list[idx].name[-5])
        dynamic = np.load(self.path_list[idx])
        dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
        return dynamic, labels


def pd_ffill(arr, axis):
    df = pd.DataFrame(arr)
    df.fillna(method='ffill', axis=axis, inplace=True)
    out = df.to_numpy()
    return out


def pd_bfill(arr, axis):
    df = pd.DataFrame(arr)
    df.fillna(method='bfill', axis=axis, inplace=True)
    out = df.to_numpy()
    return out


class FireDataset_npy(Dataset):
    def __init__(self, access_mode: str = 'spatiotemporal',
                 problem_class: str = 'classification',
                 train_val_test: str = 'train', dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1., neg_pos_ratio: float = 2.0):
        """
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2018].
                'val' gets samples from 2019.
                test' get samples from 2020
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
        self.nan_fill = nan_fill
        assert problem_class in ['classification', 'segmentation']
        if problem_class == 'classification':
            self.target = 'burned'
        else:
            self.target = 'burned_areas'
        assert self.access_mode in ['spatial', 'temporal', 'spatiotemporal']
        dataset_path = dataset_root / 'npy' / self.access_mode
        self.positives_list = list((dataset_path / 'positives').glob('*dynamic.npy'))
        self.positives_list = list(zip((self.positives_list), [1] * (len(self.positives_list))))

        num_pos = len(self.positives_list)

        self.negatives_list = list((dataset_path / 'negatives_clc').glob('*dynamic.npy'))
        self.negatives_list = list(zip(self.negatives_list, [0] * (len(self.negatives_list))))
        self.negatives_list = random.sample(self.negatives_list, int(num_pos * neg_pos_ratio))
        print(f'Positive count {len(self.positives_list)} / Negative count {len(self.negatives_list)}')

        self.path_list = self.positives_list + self.negatives_list
        val_year = '2020'
        test_year = '2021'
        self.train_path_list = [(x, y) for (x, y) in self.path_list if x.stem[:4] not in [val_year, test_year]]
        self.val_path_list = [(x, y) for (x, y) in self.path_list if x.stem[:4] == val_year]
        self.test_path_list = [(x, y) for (x, y) in self.path_list if x.stem[:4] == test_year]

        self.dynamic_idx = [i for i, feat in enumerate(variable_dict['dynamic']) if feat in self.dynamic_features]
        self.static_idx = [i for i, feat in enumerate(variable_dict['static']) if feat in self.static_features]

        if train_val_test == 'train':
            self.path_list = self.train_path_list
        elif train_val_test == 'val':
            self.path_list = self.val_path_list
        elif train_val_test == 'test':
            self.path_list = self.test_path_list
        print("#%@#$%@#$", len(self.path_list))
        random.shuffle(self.path_list)
        self.mm_dict = self._min_max_vec()

    def _min_max_vec(self):
        mm_dict = {'min': {}, 'max': {}}
        for agg in ['min', 'max']:
            if self.access_mode == 'spatial':
                mm_dict[agg]['dynamic'] = np.ones((len(self.dynamic_features), 1, 1))
                mm_dict[agg]['static'] = np.ones((len(self.static_features), 1, 1))
                for i, feat in enumerate(self.dynamic_features):
                    mm_dict[agg]['dynamic'][i, :, :] = min_max_dict[agg][self.access_mode][feat]
                for i, feat in enumerate(self.static_features):
                    mm_dict[agg]['static'][i, :, :] = min_max_dict[agg][self.access_mode][feat]

            if self.access_mode == 'temporal':
                mm_dict[agg]['dynamic'] = np.ones((1, len(self.dynamic_features)))
                mm_dict[agg]['static'] = np.ones((len(self.static_features)))
                for i, feat in enumerate(self.dynamic_features):
                    mm_dict[agg]['dynamic'][:, i] = min_max_dict[agg][self.access_mode][feat]
                for i, feat in enumerate(self.static_features):
                    mm_dict[agg]['static'][i] = min_max_dict[agg][self.access_mode][feat]

            if self.access_mode == 'spatiotemporal':
                mm_dict[agg]['dynamic'] = np.ones((1, len(self.dynamic_features), 1, 1))
                mm_dict[agg]['static'] = np.ones((len(self.static_features), 1, 1))
                for i, feat in enumerate(self.dynamic_features):
                    mm_dict[agg]['dynamic'][:, i, :, :] = min_max_dict[agg][self.access_mode][feat]
                for i, feat in enumerate(self.static_features):
                    mm_dict[agg]['static'][i, :, :] = min_max_dict[agg][self.access_mode][feat]
        return mm_dict

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, labels = self.path_list[idx]
        dynamic = np.load(path)
        static = np.load(str(path).replace('dynamic', 'static'))
        if self.access_mode == 'spatial':
            dynamic = dynamic[self.dynamic_idx]
            static = static[self.static_idx]
        elif self.access_mode == 'temporal':
            dynamic = dynamic[:, self.dynamic_idx, ...]
            static = static[self.static_idx]
        else:
            dynamic = dynamic[:, self.dynamic_idx, ...]
            static = static[self.static_idx]

        def _min_max_scaling(in_vec, max_vec, min_vec):
            return (in_vec - min_vec) / (max_vec - min_vec)

        # if self.access_mode == 'temporal':
        #     for i, feat in enumerate(self.dynamic_features):
        #         if 'LST' in feat:
        #             dynamic[:, i, ...] = pd_bfill(pd_ffill(dynamic[:, i, ...], axis=0), axis=0).reshape(-1)

        dynamic = _min_max_scaling(dynamic, self.mm_dict['max']['dynamic'], self.mm_dict['min']['dynamic'])
        static = _min_max_scaling(static, self.mm_dict['max']['static'], self.mm_dict['min']['static'])

        if self.nan_fill:
            dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
            static = np.nan_to_num(static, nan=self.nan_fill)
        return dynamic, static, 0, labels
