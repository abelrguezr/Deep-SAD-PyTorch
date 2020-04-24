from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.cicflow_dataset import CICFlowDataset
from .preprocessing import create_semisupervised_setting

import torch


class CICFlowADDataset(BaseADDataset):
    def __init__(self,
                 data_path: str,
                 n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0,
                 ratio_pollution: float = 0.0,
                 random_state=None):
        super().__init__(data_path)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0, )
        self.outlier_classes = (1, )

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1, )

        # Get train set
        train_set = CICFlowDataset(data_path=self.root,
                                   train=True,
                                   random_state=random_state)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(
            train_set.targets.cpu().data.numpy(), self.normal_classes,
            self.outlier_classes, self.known_outlier_classes,
            ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(
            semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        # self.train_set = Subset(train_set, idx)
        self.train_set = train_set


        # Get test set
        self.test_set = CICFlowDataset(data_path=self.root,
                                       train=False,
                                       random_state=random_state)

    def loaders(self,
                batch_size: int,
                shuffle_train=False,
                shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True)
        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 drop_last=False)
        return train_loader, test_loader
