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

dataset_path = Path.home() / 'jh-shared/iprapas/uc3'
root = dataset_path / 'image_classification'

dynamic_features = [
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

static_features = ['dem_mean',
                   'dem_std',
                   'aspect_mean',
                   'aspect_std',
                   'slope_mean',
                   'slope_std',
                   'roads_density_2020',
                   'population_density']

target = 'burned_areas'


class GreeceFireDataset(Dataset):
    def __init__(self, data_dir=root, mode='lstm', train=True, sel_dynamic_features=dynamic_features,
                 sel_static_features=static_features):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dynamic_file_paths = sorted(
            [os.path.join(data_dir, 'dynamic', x) for x in os.listdir(data_dir / 'dynamic')],
            key=os.path.getmtime)[1:]
        self.static_file_paths = sorted([os.path.join(data_dir, 'static', x) for x in os.listdir(data_dir / 'static')],
                                        key=os.path.getmtime)
        self.num_static_features = len(sel_static_features)
        self.num_dynamic_features = len(sel_dynamic_features)
        self.static_features_idx = [i for i, x in enumerate(static_features) if x in sel_static_features]
        self.dynamic_features_idx = [i for i, x in enumerate(dynamic_features) if x in sel_dynamic_features]
        self.train = train
        if self.train:
            self.dynamic_file_paths = [x for x in self.dynamic_file_paths if '2020' not in x]
            self.static_file_paths = [x for x in self.static_file_paths if '2020' not in x]
        else:
            self.dynamic_file_paths = [x for x in self.dynamic_file_paths if '2020' in x]
            self.static_file_paths = [x for x in self.static_file_paths if '2020' in x]
        with open(dataset_path / 'image_classification' / 'labels' / 'predictions.csv', newline='\n') as f:
            reader = csv.reader(f)
            if self.train:
                self.labels = [int(row[0]) for row in reader][:len(self.static_file_paths)]
            else:
                self.labels = [int(row[0]) for row in reader][-len(self.static_file_paths):]
        self.mode = mode
        assert mode in ['conv', 'lstm', 'convlstm']
        if mode == 'conv':
            mean_dynamic = [7.5731e-01, 1.6864e+02, 1.4852e+02, 4.2403e+07, 7.8031e+02,
                            7.4666e-01, -6.4321e-02, 2.4285e+02, -1.7904e-01, -1.1165e+00,
                            -1.6475e+00, 2.3514e+02, -1.7993e-01]
            std_dynamic = [8.9428e-01, 1.5158e+02, 1.4564e+02, 2.6030e+07, 1.3837e+03, 1.6735e+00,
                           1.7839e+00, 1.1584e+02, 3.8474e-01, 1.3258e+00, 1.8643e+00, 1.1215e+02,
                           3.8430e-01]
            mean_static = [411.7267, 41.5362, 149.0853, 74.0876, 9.1543, 4.3644, 3.9257,
                           47.2859]
            std_static = [423.7853, 40.3595, 77.9320, 40.4638, 8.2169, 3.0958, 10.5679,
                          540.9679]
        elif mode == 'lstm':
            mean_dynamic = [4.3142e-01, 2.0252e+02, 1.7790e+02, 4.9290e+07, 3.0420e+02,
                            9.7633e-01, 3.3601e-02, 2.6125e+02, -1.2039e-01, -1.0969e+00,
                            -1.7031e+00, 2.5326e+02, -1.2132e-01]
            std_dynamic = [4.7098e-01, 1.4393e+02, 1.4210e+02, 1.9732e+07, 9.2947e+02, 1.7244e+00,
                           1.8966e+00, 9.7741e+01, 3.2709e-01, 1.4055e+00, 1.9181e+00, 9.4733e+01,
                           3.2673e-01]
            mean_static = [455.2351, 51.4826, 177.8961, 86.1206, 11.4065, 5.3960, 4.6404,
                           26.2571]
            std_static = [392.2036, 37.9239, 49.0176, 28.8770, 7.4689, 2.5952, 8.7741,
                          244.3964]
        elif mode == 'convlstm':
            mean_dynamic = [7.5746e-01, 1.6587e+02, 1.4495e+02, 4.2414e+07, 7.8837e+02,
                            7.7565e-01, -6.5821e-03, 2.4260e+02, -1.7903e-01, -1.0847e+00,
                            -1.5780e+00, 2.3501e+02, -1.7992e-01]
            std_dynamic = [8.9433e-01, 1.5180e+02, 1.4559e+02, 2.6051e+07, 1.3885e+03, 1.6483e+00,
                           1.7267e+00, 1.1572e+02, 3.8474e-01, 1.2877e+00, 1.7475e+00, 1.1209e+02,
                           3.8431e-01]
            mean_static = [411.7267, 41.5362, 149.0853, 74.0876, 9.1543, 4.3644, 3.9257,
                           47.2859]
            std_static = [423.7853, 40.3595, 77.9320, 40.4638, 8.2169, 3.0958, 10.5679,
                          540.9679]
        else:
            raise Exception('Mode not in available modes [conv, lstm, convlstm]')
        mean_static_ = [mean_static[i] for i in self.static_features_idx]
        mean_dynamic_ = [mean_dynamic[i] for i in self.dynamic_features_idx]
        std_static_ = [std_static[i] for i in self.static_features_idx]
        std_dynamic_ = [std_dynamic[i] for i in self.dynamic_features_idx]
        self.dynamic_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(tuple(mean_dynamic_), tuple(std_dynamic_))]
        )
        self.static_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(tuple(mean_static_), tuple(std_static_))]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        if self.mode == 'conv':
            dynamic = np.load(self.dynamic_file_paths[idx])[-1]
            dynamic = dynamic[self.dynamic_features_idx, ...]
            static = np.load(self.static_file_paths[idx])[:-1]
            static = static[self.static_features_idx, ...]
            clc = np.load(self.static_file_paths[idx])[-1].astype('int8')
            clc = (np.arange(45) == clc[..., None]).astype(int)
            clc = np.transpose(clc, (2, 0, 1))
        elif self.mode == 'lstm':
            dynamic = np.load(self.dynamic_file_paths[idx])
            dynamic = dynamic[:, :, dynamic.shape[2] // 2, dynamic.shape[3] // 2]
            dynamic = dynamic[:, self.dynamic_features_idx, ...]
            static = np.load(self.static_file_paths[idx])[:-1]
            static = static[self.static_features_idx, ...]
            clc = np.load(self.static_file_paths[idx])[-1]
            static = static[:, static.shape[1] // 2, static.shape[2] // 2]
            clc_perm = clc[clc.shape[0] // 2, clc.shape[1] // 2].astype('int8')
            clc = np.zeros(45)
            clc[clc_perm] = 1
        # convlstm
        else:
            dynamic = np.load(self.dynamic_file_paths[idx])
            dynamic = dynamic[:, self.dynamic_features_idx, ...]
            static = np.load(self.static_file_paths[idx])
            static = static[self.static_features_idx, ...]
            clc = np.load(self.static_file_paths[idx])[-1].astype('int8')
            clc = (np.arange(45) == clc[..., None]).astype(int)
            clc = np.transpose(clc, (2, 0, 1))
        dynamic = np.nan_to_num(dynamic, nan=-1.)
        static = np.nan_to_num(static, nan=-1.)
        clc = np.nan_to_num(clc, nan=-1.)
        # dynamic = np.expand_dims(dynamic, axis=0)
        # static = np.expand_dims(static, axis=0)
        dynamic = np.transpose(dynamic, (2, 1, 0))
        static = np.transpose(static, (2, 1, 0))
        dynamic = self.dynamic_transforms(dynamic)
        static = self.static_transforms(static)
        return dynamic, static, clc, labels
