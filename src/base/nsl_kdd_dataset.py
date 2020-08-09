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


class NSLKDDDataset(Dataset):
    def __init__(self,
                 root: str,
                 idx=None,
                 train=True,
                 random_state=None,
                 download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            data_path = os.path.expanduser(root)

        self.root = os.path.abspath(Path(root))
        self.train = train  # training set or test set
        X, y = self._get_csv_data(train, idx)

        # if download:
        #     self.download()

        idx_norm = np.invert(y)
        idx_out = y

        X_norm, y_norm = X[idx_norm], y[idx_norm]

        X_out, y_out = X[idx_out], y[idx_out]

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X)
        X_stand = scaler.transform(X)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_stand)
        X_scaled = minmax_scaler.transform(X_stand)

        self.data = torch.tensor(X_scaled, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)

    def _get_csv_data(self, train=True, idx=None):

        if train: path = self.root + '/KDDTrain+.csv'
        else: path = self.root + '/KDDTest+.csv'

        header_df = pd.read_csv(self.root + '/Field Names.csv', header=None)
        headers = header_df.iloc[:, 0].to_list() + ['label', 'unknown']
        df = pd.read_csv(path, names=headers).drop(

            ['protocol_type', 'service', 'flag'], axis=1)

        X = df.iloc[:, :-2].to_numpy()
        y = np.array(df['label'] != 'normal')
        if (idx is not None) and train:
            X = X[idx]
            y = y[idx]

        return X, y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(
            self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)