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


class CICFlowDataset(Dataset):
    """
    CICFlowDataset class for datasets from CICFlowMeter (https://github.com/ahlashkari/CICFlowMeter) feature extractor.

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """
    def __init__(self,
                 root: str,
                 train_dates=None,
                 test_dates=None,
                 train=True,
                 random_state=None,
                 split=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            data_path = os.path.expanduser(root)

        self.root = os.path.abspath(Path(root))
        self.train = train  # training set or test set
        self.train_dates = train_dates  # training set or test set
        self.test_dates = test_dates  # training set or test set

        X_train, y_train = self._get_csv_data(train_dates)

        if self.train:
            X, y = X_train, y_train
        else:
            X, y = self._get_csv_data(test_dates)

        idx_norm = np.invert(y)
        idx_out = y

        X_norm, y_norm = X[idx_norm], y[idx_norm]

        X_out, y_out = X[idx_out], y[idx_out]

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X = scaler.transform(X)
        X_train = scaler.transform(X_train)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train)
        X = minmax_scaler.transform(X)

        if split:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.3, random_state=42)

            if self.train:
                X, y = X_train, y_train
            else:
                X, y = X_test, y_test

        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)

    def _get_csv_data(self, dates):

        date_str = [str(date).split()[0] for date in dates]

        X_dates_paths = [
            self.root + '/' + date.replace('-', '') + '/merged_flows_flow.npy'
            for date in date_str
        ]
        y_dates_paths = [
            self.root + '/' + date.replace('-', '') + '/merged_flows_label.npy'
            for date in date_str
        ]

        X = np.ma.row_stack(tuple(np.load(path) for path in X_dates_paths))
        y = np.concatenate(tuple(np.load(path) for path in y_dates_paths))

        # drop NaNs
        # X = X[~np.isnan(X)]
        # y = y[~np.isnan(X)]

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

    def _get_cicflow_dtypes(self):

        dtypes = {
            'Flow ID': 'object',
            'Src IP': 'object',
            'Src Port': 'object',
            'Dst IP': 'object',
            'Dst Port': 'object',
            'Protocol': 'object',
            'Flow Duration': 'int64',
            'Tot Fwd Pkts': 'int64',
            'Tot Bwd Pkts': 'int64',
            'TotLen Fwd Pkts': 'float64',
            'TotLen Bwd Pkts': 'float64',
            'Fwd Pkt Len Max': 'float64',
            'Fwd Pkt Len Min': 'float64',
            'Fwd Pkt Len Mean': 'float64',
            'Fwd Pkt Len Std': 'float64',
            'Bwd Pkt Len Max': 'float64',
            'Bwd Pkt Len Min': 'float64',
            'Bwd Pkt Len Mean': 'float64',
            'Bwd Pkt Len Std': 'float64',
            'Flow Byts/s': 'float64',
            'Flow Pkts/s': 'float64',
            'Flow IAT Mean': 'float64',
            'Flow IAT Std': 'float64',
            'Flow IAT Max': 'float64',
            'Flow IAT Min': 'float64',
            'Fwd IAT Tot': 'float64',
            'Fwd IAT Mean': 'float64',
            'Fwd IAT Std': 'float64',
            'Fwd IAT Max': 'float64',
            'Fwd IAT Min': 'float64',
            'Bwd IAT Tot': 'float64',
            'Bwd IAT Mean': 'float64',
            'Bwd IAT Std': 'float64',
            'Bwd IAT Max': 'float64',
            'Bwd IAT Min': 'float64',
            'Fwd PSH Flags': 'int64',
            'Bwd PSH Flags': 'int64',
            'Fwd URG Flags': 'int64',
            'Bwd URG Flags': 'int64',
            'Fwd Header Len': 'int64',
            'Bwd Header Len': 'int64',
            'Fwd Pkts/s': 'float64',
            'Bwd Pkts/s': 'float64',
            'Pkt Len Min': 'float64',
            'Pkt Len Max': 'float64',
            'Pkt Len Mean': 'float64',
            'Pkt Len Std': 'float64',
            'Pkt Len Var': 'float64',
            'FIN Flag Cnt': 'int64',
            'SYN Flag Cnt': 'int64',
            'RST Flag Cnt': 'int64',
            'PSH Flag Cnt': 'int64',
            'ACK Flag Cnt': 'int64',
            'URG Flag Cnt': 'int64',
            'CWE Flag Count': 'int64',
            'ECE Flag Cnt': 'int64',
            'Down/Up Ratio': 'float64',
            'Pkt Size Avg': 'float64',
            'Fwd Seg Size Avg': 'float64',
            'Bwd Seg Size Avg': 'float64',
            'Fwd Byts/b Avg': 'int64',
            'Fwd Pkts/b Avg': 'int64',
            'Fwd Blk Rate Avg': 'int64',
            'Bwd Byts/b Avg': 'int64',
            'Bwd Pkts/b Avg': 'int64',
            'Bwd Blk Rate Avg': 'int64',
            'Subflow Fwd Pkts': 'int64',
            'Subflow Fwd Byts': 'int64',
            'Subflow Bwd Pkts': 'int64',
            'Subflow Bwd Byts': 'int64',
            'Init Fwd Win Byts': 'int64',
            'Init Bwd Win Byts': 'int64',
            'Fwd Act Data Pkts': 'int64',
            'Fwd Seg Size Min': 'int64',
            'Active Mean': 'int64',
            'Active Std': 'int64',
            'Active Max': 'int64',
            'Active Min': 'int64',
            'Idle Mean': 'float64',
            'Idle Std': 'float64',
            'Idle Max': 'float64',
            'Idle Min': 'float64',
            'Label': 'object',
            'anomaly_id': 'object',
            'nb_anomaly_fields_missing': 'int64',
            'packet_entry_id': 'object'
        }

        return dtypes
