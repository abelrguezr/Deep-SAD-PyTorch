import json
import torch
from functools import partial
from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer
from base.base_nn_model import BaseNNModel


class DeepSAD(BaseNNModel):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).

    """
    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        super().__init__(eta=self.eta)

    def train(self,
              dataset: BaseADDataset,
              optimizer_name: str = 'adam',
              lr: float = 0.001,
              n_epochs: int = 50,
              lr_milestones: tuple = (),
              batch_size: int = 128,
              weight_decay: float = 1e-6,
              device: str = 'cuda',
              n_jobs_dataloader: int = 0,
              reporter=None):
        """Trains the Deep SAD model on the training data."""

        trainer = DeepSADTrainer(self.c,
                                 self.eta,
                                 optimizer_name=optimizer_name,
                                 lr=lr,
                                 n_epochs=n_epochs,
                                 lr_milestones=lr_milestones,
                                 batch_size=batch_size,
                                 weight_decay=weight_decay,
                                 device=device,
                                 n_jobs_dataloader=n_jobs_dataloader,
                                 reporter=reporter)

        self._train(trainer, dataset)

        return self

    def test(self,
             dataset: BaseADDataset,
             device: str = 'cuda',
             n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c,
                                          self.eta,
                                          device=device,
                                          n_jobs_dataloader=n_jobs_dataloader)

        self._test(self.trainer, dataset)

        return self
