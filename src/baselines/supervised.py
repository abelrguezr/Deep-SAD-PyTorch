import json
import torch
from functools import partial
from base.base_dataset import BaseADDataset
from base.base_nn_model import BaseNNModel
from networks.main import build_network, build_autoencoder
from optim.supervised_trainer import SupervisedTrainer
from optim.ae_trainer import AETrainer


class Supervised(BaseNNModel):
    """A class for a supervised model.

    Attributes:
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: Trainer to train a model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """
    def __init__(self):
        super().__init__()

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

        trainer = SupervisedTrainer(optimizer_name=optimizer_name,
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
            self.trainer = SupervisedTrainer(
                device=device, n_jobs_dataloader=n_jobs_dataloader)

        self._test(self.trainer, dataset)

        return self
