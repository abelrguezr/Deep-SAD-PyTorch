from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url
import pandas as pd
import os
import torch
import numpy as np


class UGR16Dataset(Dataset):
    """
    UGR16Dataset class for datasets from UGR'16: A New Dataset for the Evaluation of Cyclostationarity-Based Network IDSs (UGR16): https://nesg.ugr.es/nesg-ugr16/index.php

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    urls = {
        'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1',
        'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1',
        'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1',
        'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=1',
        'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1',
        'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1'
    }

    def __init__(self, root: str, csv_file: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(csv_file, torch._six.string_classes):
            csv_file = os.path.expanduser(csv_file)

        self.root = Path(root)
        self.csv_file = csv_file
        self.train = train  # training set or test set

        self.train = train  # training set or test set
        self.data_file = self.root / self.csv_file

        # if download:
        #     self.download()


        headers = ['timestamp_end','duration','src_addr','dest_addr','src_port','dest_port','protocol','flags','forwarding_status','service_type','packets','bytes','label']

        df_chunk = pd.read_csv(self.data_file,nrows=1100000, names=headers)

        outliers = df_chunk['label'] != 'background'


        # X = df_chunk.loc[:, df_chunk.columns != 'label']
        # y = df_chunk['label'].ravel()
        y = outliers.values
        X = df_chunk[['packets','bytes']].values

        idx_norm = np.invert(y)
        idx_out = y

        # 80% data for training and 20% for testing; keep outlier ratio
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],             test_size=0.2, shuffle=False)


        X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
        test_size=0.2, shuffle=False)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        # scaler = StandardScaler().fit(X_train)
        # X_train_stand = scaler.transform(X_train)
        # X_test_stand = scaler.transform(X_test)

        # # Scale to range [0,1]
        # minmax_scaler = MinMaxScaler().fit(X_train_stand)
        # X_train_scaled = minmax_scaler.transform(X_train_stand)
        # X_test_scaled = minmax_scaler.transform(X_test_stand)

        # Unscaled data for testing
        X_train_stand = X_train
        X_test_stand = X_test
        X_train_scaled = X_train_stand
        X_test_scaled = X_test_stand

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)

    # def download(self):
    #     """Download the ODDS dataset if it doesn't exist in root already."""

    #     if self._check_exists():
    #         return

    #     # download file
    #     download_url(self.urls[self.dataset_name], self.csv_file, self.file_name)

    #     print('Done!')
