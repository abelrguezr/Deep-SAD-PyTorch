import click
import os
import pandas as pd
import torch
import logging
import random
import numpy as np
import logging
import ray
import pickle
from sklearn.metrics import precision_recall_fscore_support
from ray import tune
from ray.tune import track
from baselines.supervised import Supervised
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
from datasets.cicflow import CICFlowADDataset
from networks.mlp import MLP
from utils.misc import get_train_val_split
from models.deepSVDD import DeepSVDD
from datasets.main import load_dataset
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler


class SupervisedCICFlowExp(tune.Trainable):
    def _setup(self, cfg):
        # self.training_iteration = 0
        self.test_labels = None
        self.val_labels = None
        self.val_scores = None
        self.test_scores = None

        trial_idx = cfg['__trial_index__']
        train, val = cfg['train_dates'][trial_idx]
        test = cfg['test_dates']

        self.dataset = CICFlowADDataset(root=os.path.abspath(cfg['data_path']),
                                        n_known_outlier_classes=1,
                                        train_dates=cfg['period'][train],
                                        val_dates=cfg['period'][val],
                                        test_dates=test,
                                        shuffle=True)

        self.model = Supervised()
        self.model.set_trainer(optimizer_name=cfg['optimizer_name'],
                               lr=cfg['lr'],
                               n_epochs=cfg['n_epochs'],
                               lr_milestones=cfg['lr_milestone'],
                               batch_size=cfg['batch_size'],
                               weight_decay=cfg['weight_decay'],
                               device=cfg['device'],
                               n_jobs_dataloader=cfg["n_jobs_dataloader"])
        self.model.setup(self.dataset, cfg['net_name'])

    def _train(self):
        self.model.train_one_step(self.training_iteration)
        self.model.test(self.dataset, val=True)
        self.model.test(self.dataset, val=False)

        val_labels, val_scores, _ = self.model.trainer.get_results("val")
        test_labels, test_scores, _ = self.model.trainer.get_results("test")

        results = locals().copy()
        del results["self"]

        self.results = results

        res = {
            phase +'_' +metric: result
            for phase in ["val", "test"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
            for pr, rec, f1, sup in
            [precision_recall_fscore_support(labels, scores)]
            for metric, result in zip(*[('pr', 'rec', 'f1',
                                         'sup'), (pr, rec, f1, sup)])
        }

        return {**res}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir,
                                       str(self.trial_id) + "_model.pth")
        self.model.save_model(checkpoint_path)
        pickle.dump(self.results,
                    open(os.path.join(checkpoint_dir, 'results.pkl'), "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_model(checkpoint_path)


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_model',
              type=click.Path(exists=True),
              default=None,
              help='Model file path (default: None).')
@click.option('--experiment_path',
              type=click.Path(exists=True),
              default='~/ray_results',
              help='Model file path (default: None).')
@click.option('--ratio_known_normal',
              type=float,
              default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier',
              type=float,
              default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--seed',
              type=int,
              default=0,
              help='Set seed. If -1, use randomization.')
@click.option(
    '--optimizer_name',
    type=click.Choice(['adam']),
    default='adam',
    help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--validation',
              type=click.Choice(['kfold', 'time_series', 'index']),
              default='kfold',
              help='Validation strategy.')
@click.option(
    '--lr',
    type=float,
    default=0.001,
    help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--n_epochs',
              type=int,
              default=50,
              help='Number of epochs to train.')
@click.option(
    '--lr_milestone',
    type=int,
    default=0,
    multiple=True,
    help=
    'Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.'
)
@click.option('--batch_size',
              type=int,
              default=128,
              help='Batch size for mini-batch training.')
@click.option(
    '--weight_decay',
    type=float,
    default=1e-6,
    help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain',
              type=bool,
              default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name',
              type=click.Choice(['adam']),
              default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option(
    '--ae_lr',
    type=float,
    default=0.001,
    help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs',
              type=int,
              default=100,
              help='Number of epochs to train autoencoder.')
@click.option(
    '--ae_lr_milestone',
    type=int,
    default=0,
    multiple=True,
    help=
    'Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.'
)
@click.option('--ae_batch_size',
              type=int,
              default=128,
              help='Batch size for mini-batch autoencoder training.')
@click.option(
    '--ae_weight_decay',
    type=float,
    default=1e-6,
    help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option(
    '--num_threads',
    type=int,
    default=0,
    help=
    'Number of threads used for parallelizing CPU operations. 0 means that all resources are used.'
)
@click.option('--validation',
              type=click.Choice(['kfold', 'time_series', 'index']),
              default='index',
              help='Validation strategy.')
@click.option(
    '--n_jobs_dataloader',
    type=int,
    default=0,
    help=
    'Number of workers for data loading. 0 means that the data will be loaded in the main process.'
)
@click.option(
    '--normal_class',
    type=int,
    default=0,
    help=
    'Specify the normal class of the dataset (all other classes are considered anomalous).'
)
@click.option(
    '--known_outlier_class',
    type=int,
    default=1,
    help=
    'Specify the known outlier class of the dataset for semi-supervised anomaly detection.'
)
@click.option(
    '--n_known_outlier_classes',
    type=int,
    default=0,
    help='Number of known outlier classes.'
    'If 0, no anomalies are known.'
    'If 1, outlier class as specified in --known_outlier_class option.'
    'If > 1, the specified number of outlier classes will be sampled at random.'
)
def main(data_path, experiment_path, load_model, ratio_known_normal,
         ratio_known_outlier, seed, optimizer_name, validation, lr, n_epochs,
         lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name,
         ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class,
         n_known_outlier_classes):
    

    ray.init(address='auto')

    data_path = os.path.abspath(data_path)
    n_splits = 4

    period = np.array([
        '2019-11-08', '2019-11-09', '2019-11-11', '2019-11-12', '2019-11-13',
        '2019-11-14', '2019-11-15'
    ])

    test_dates = period[-2:]
    train_dates = get_train_val_split(period[:-2], validation, n_splits)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_config = {
        **locals().copy(),
        'net_name': 'cicflow_mlp',
    }

    if exp_config['seed'] != -1:
        random.seed(exp_config['seed'])
        np.random.seed(exp_config['seed'])
        torch.manual_seed(exp_config['seed'])
        torch.cuda.manual_seed(exp_config['seed'])
        torch.backends.cudnn.deterministic = True

    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(
        name="SupervisedCICFlowExp",
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-6, 0.4],
                "log_scale": True
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [1e-6, 1.0],
                "log_scale": True
            },
        ],
        objective_name="val_f1",
    )

    search_alg = AxSearch(ax)
    re_search_alg = Repeater(search_alg, repeat=n_splits)

    sched = ASHAScheduler(time_attr='training_iteration',
                          grace_period=10,
                          metric="val_f1")

    analysis = tune.run(SupervisedCICFlowExp,
                        name="SupervisedCICFlowExp",
                        checkpoint_at_end=True,
                        checkpoint_freq=5,
                        stop={
                            "training_iteration": 100,
                        },
                        resources_per_trial={"gpu": 1},
                        num_samples=30,
                        local_dir=experiment_path,
                        search_alg=re_search_alg,
                        scheduler=sched,
                        config=exp_config)

    print("Best config is:", analysis.get_best_config(metric="val_f1"))



if __name__ == '__main__':
    main()