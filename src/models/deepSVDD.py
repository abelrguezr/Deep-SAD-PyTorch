import json
import torch
from base.base_nn_model import BaseNNModel
from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.deepSVDD_trainer import DeepSVDDTrainer
from optim.ae_trainer import AETrainer


class DeepSVDD(BaseNNModel):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        super().__init__(objective=objective, nu=nu, c=None,R=0.0)



    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, 
              reporter = None):
        """Trains the Deep SVDD model on the training data."""

        self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader,
                                       reporter = reporter)
        # Get the model
        self._train(self.trainer, dataset)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list


        return self

    def set_trainer(self,
              optimizer_name: str = 'adam',
              lr: float = 0.001,
              n_epochs: int = 50,
              lr_milestones: tuple = (),
              batch_size: int = 128,
              weight_decay: float = 1e-6,
              device: str = 'cuda',
              n_jobs_dataloader: int = 0,
              reporter=None):

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader,
                                       reporter = reporter)  

        return self.trainer                                    

    def train_one_step(self, dataset,epoch: int = 0):
        self._train_one_step(self.trainer, dataset, epoch)
        return self    

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self._test(self.trainer, dataset)
        # Get results
        return self