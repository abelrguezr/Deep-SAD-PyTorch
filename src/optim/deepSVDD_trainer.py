from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class DeepSVDDTrainer(BaseTrainer):
    def __init__(self,
                 objective,
                 R,
                 c,
                 nu: float,
                 optimizer_name: str = 'adam',
                 lr: float = 0.001,
                 n_epochs: int = 150,
                 lr_milestones: tuple = (),
                 batch_size: int = 128,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0,
                 reporter=None):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones,
                         batch_size, weight_decay, device, n_jobs_dataloader)

        assert objective in (
            'one-class', 'soft-boundary'
        ), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(
            R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        self.train_loader = None

        # Results
        self.train_time = None
        self.test_labels = None
        self.val_labels = None
        self.val_scores = None
        self.train_labels = None
        self.train_scores = None
        self.test_loss = None
        self.val_loss = None
        self.test_scores = None
        self.reporter = reporter

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        try:
            train_loader, _, _ = dataset.loaders(
                batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        except:
            train_loader, _ = dataset.loaders(
                batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' %
                            float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c)**2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R**2
                    loss = self.R**2 + (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (
                        epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu),
                                               device=self.device)

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(
                epoch + 1, self.n_epochs, epoch_train_time,
                epoch_loss / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def setup(self, dataset, net):
        logger = logging.getLogger()
        net = net.to(self.device)

        if self.train_loader is None:
            try:
                self.train_loader, _, _ = dataset.loaders(
                    batch_size=self.batch_size,
                    num_workers=self.n_jobs_dataloader)
            except:
                self.train_loader, _ = dataset.loaders(
                    batch_size=self.batch_size,
                    num_workers=self.n_jobs_dataloader)
        # Set optimizer (Adam optimizer for now)
        self.optimizer = optim.Adam(net.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(self.train_loader, net)
            logger.info('Center c initialized.')
                
                
        return net
    

    def train_one_step(self, net: BaseNet, epoch: int):
        logger = logging.getLogger()

        # Set device for network
        # net = net.to(self.device)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        if (True):
            self.scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' %
                            float(self.scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in self.train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                self.optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c)**2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R**2
                    loss = self.R**2 + (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                self.optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (
                        epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu),
                                               device=self.device)

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(
                epoch + 1, self.n_epochs, epoch_train_time,
                epoch_loss / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, set_split="train"):
        if set_split == "train":
            try:
                train_loader, _, _ = dataset.loaders(
                    batch_size=self.batch_size,
                    num_workers=self.n_jobs_dataloader)

            except:
                train_loader, _ = dataset.loaders(
                    batch_size=self.batch_size,
                    num_workers=self.n_jobs_dataloader)
            self.train_labels, self.train_scores, self.train_loss = self._test(
                train_loader, net)

        elif set_split == "val":
            try:
                _, val_loader, _ = dataset.loaders(
                    batch_size=self.batch_size,
                    num_workers=self.n_jobs_dataloader)

                self.val_labels, self.val_scores, self.val_loss = self._test(
                    val_loader, net)
            except:
                raise ValueError(
                    "The dataset does not support validation DataLoader")
        else:
            try:
                _, _, test_loader = dataset.loaders(
                    batch_size=self.batch_size,
                    num_workers=self.n_jobs_dataloader)
            except:
                _, test_loader = dataset.loaders(
                    batch_size=self.batch_size,
                    num_workers=self.n_jobs_dataloader)

            self.test_labels, self.test_scores, self.test_loss = self._test(
                test_loader, net)

    def get_results(self, phase='val'):
        if phase == 'val':
            return self.val_labels, self.val_scores, self.val_loss
        elif phase == 'train':
            return self.train_labels, self.train_scores, self.train_loss
        else:
            return self.test_labels, self.test_scores, self.test_loss

    def _test(self, loader, net: BaseNet):
        logger = logging.getLogger()
        epoch_loss = 0.0
        n_batches = 0

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels, idx, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c)**2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R**2
                    loss = self.R**2 + (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                    scores = dist

                epoch_loss += loss.item()
                n_batches += 1
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(
                    zip(idx.cpu().data.numpy().tolist(),
                        labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        test_loss = epoch_loss / n_batches

        return labels, scores, test_loss

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
