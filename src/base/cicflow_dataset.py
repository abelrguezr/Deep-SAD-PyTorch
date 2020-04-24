from pathlib import Path
from torch.utils.data import IterableDataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url
import pandas as pd
import itertools
import os
import torch
import numpy as np


class CICFlowDataset(IterableDataset):
    """
    CICFlowDataset class for datasets from CICFlowMeter (https://github.com/ahlashkari/CICFlowMeter) feature extractor.

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """
    def __init__(self,
                 data_path: str,
                 train=True,
                 random_state=None,
                 download=False):
        super(IterableDataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(data_path, torch._six.string_classes):
            data_path = os.path.expanduser(data_path)

        self.data_path = Path(data_path)
        self.train = train  # training set or test set

        self.train = train  # training set or test set

        # if download:
        #     self.download()
        X, y = self._get_csv_data()

        idx_norm = np.invert(y)
        idx_out = y

        # 80% data for training and 20% for testing; keep outlier ratio
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
            X[idx_norm], y[idx_norm], test_size=0.2, shuffle=False)

        X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(
            X[idx_out], y[idx_out], test_size=0.2, shuffle=False)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            shuffle=False)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)

    def _get_csv_data(self):

        # Load csv containing the data and labeled anomalous flows
        df_flows = pd.read_csv(self.data_path,
                               dtype=self._get_cicflow_dtypes(),
                               index_col='Timestamp').dropna()

        # Label anomalous packets as such and everything else as background
        outliers = (df_flows['Label'].values == 'anomalous') | (
            df_flows['Label'].values == 'suspicious')

        y = outliers.astype(int)
        X = df_flows \
            .iloc[:,6:-4] \
            .to_numpy()

        return X, y

    def __iter__(self):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target, idx = iter(self.data), iter(
            self.targets), iter(self.semi_targets), itertools.count()


        return zip(sample, target, semi_target, idx)



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
