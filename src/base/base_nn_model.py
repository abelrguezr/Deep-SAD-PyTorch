import json
import torch
from functools import partial
from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.ae_trainer import AETrainer


class BaseNNModel(object):
    """
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
    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)
        self.kwargs = kwargs

        self.net_name = None
        self.net = None  # neural network phi
        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None
        self.trainer = None

        self.results = {
            'train_time': None,
            'auc_roc': None,
            'auc_pr': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'auc_roc': None,
            'test_time': None
        }

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

        return self

    def _train(self, trainer, dataset):
        """Trains the model on the training data."""

        self.trainer = trainer

        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time

        return self
    
    def train_one_step(self, epoch):
        """Trains the model on the training data."""

        # Get the model
        self.train_loss = self.trainer.train_one_step(self.net, epoch)
        self.results['train_time'] = self.trainer.train_time

        return self

    def setup(self,dataset, net_name):
        self.set_network(net_name)
        self.trainer.setup(dataset, self.net)

        return self    

    def _test(self, trainer, dataset):
        """Tests the Deep SAD model on the test data."""

        self.trainer = trainer
        self.trainer.test(dataset, self.net)

        # Get results
        self.results['auc_roc'] = self.trainer.auc_roc
        self.results['auc_pr'] = self.trainer.auc_pr
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

        return self

    def pretrain(self,
                 dataset: BaseADDataset,
                 optimizer_name: str = 'adam',
                 lr: float = 0.001,
                 n_epochs: int = 100,
                 lr_milestones: tuple = (),
                 batch_size: int = 128,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name,
                                    lr=lr,
                                    n_epochs=n_epochs,
                                    lr_milestones=lr_milestones,
                                    batch_size=batch_size,
                                    weight_decay=weight_decay,
                                    device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        self.ae_results['auc_roc'] = self.ae_trainer.auc_roc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

        return self

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

        return self

    def save_model(self, export_model_path, save_ae=False):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save(
            {
                'net_dict': net_dict,
                'ae_net_dict': ae_net_dict,
                **self.kwargs
            }, export_model_path)

        return self

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        for key in self.kwargs:
            setattr(self, key, model_dict[key])

        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

        return self

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)
