from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
import logging
import time
import torch
import torch.optim as optim
import numpy as np


class SupervisedTrainer(BaseTrainer):
    def __init__(self,
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

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.auc_roc = None
        self.test_time = None
        self.test_scores = None
        self.reporter = reporter

    def train(self, dataset: BaseADDataset, net: BaseNet):

        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size,
                                          num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Set loss function
        criterion = BCEWithLogitsLoss()
  

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            for data in train_loader:
                inputs, targets , _, _ = data
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                targets=targets.type_as(outputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' %
                            float(scheduler.get_lr()[0]))

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(
                f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                f'| Train Loss: {epoch_loss / n_batches:.6f} |')

            if self.reporter:
                self._log_train(net, dataset)
                self.reporter(
                    **
                    {'train/loss/' + str(dataset.id): epoch_loss / n_batches})

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def train_one_step(self, dataset: BaseADDataset, net: BaseNet, epoch: int):

        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size,
                                          num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Set loss function
        criterion = BCEWithLogitsLoss()
  
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        
        epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
            
        for data in train_loader:
            inputs, targets , _, _ = data
            inputs, targets = inputs.to(self.device), targets.to(
                self.device)

            # Zero the network parameter gradients
            optimizer.zero_grad()

            # Update network parameters via backpropagation: forward + backward + optimize
            outputs = net(inputs)
            targets=targets.type_as(outputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            scheduler.step()

        if epoch in self.lr_milestones:
            logger.info('  LR scheduler: new learning rate is %g' %
                        float(scheduler.get_lr()[0]))

            # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        logger.info(
                f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                f'| Train Loss: {epoch_loss / n_batches:.6f} |')

            
        return {'train_loss': epoch_loss / n_batches}



    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size,
                                         num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)
        criterion = BCEWithLogitsLoss()


        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                labels=labels.type_as(outputs)
                loss = criterion(outputs, labels.unsqueeze(1))

                scores = outputs.sigmoid()

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(
                    zip(idx.cpu().data.numpy().tolist(),
                        labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute metrics
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        # AUC
        self.auc_roc = roc_auc_score(labels, scores)
        # PR-curve
        self.pr_curve = precision_recall_curve(labels, scores)
        precision, recall, thresholds = self.pr_curve
        self.auc_pr = auc(recall, precision)
        self.test_loss = epoch_loss / n_batches

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC-ROC: {:.2f}%'.format(100. * self.auc_roc))
        logger.info('Test AUC-PR: {:.2f}%'.format(100. * self.auc_pr))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

    def _log_train(self, net, dataset):

        id_data = str(dataset.id)
        self.test(dataset, net)

        self.reporter(
            **{
                'test/auc_roc/' + id_data: self.auc_roc,
                'test/auc_pr/' + id_data: self.auc_pr,
                'test/loss/' + id_data: self.test_loss
            })
