import click
import os
import pandas as pd
import torch
import logging
import random
import numpy as np
import logging
import ray
from ray import tune
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from baselines.isoforest import IsoForest
from ray.tune import track
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
from sklearn.model_selection import TimeSeriesSplit, KFold, train_test_split
from datasets.cicflow import CICFlowADDataset
from models.deepSVDD import DeepSVDD
from datasets.main import load_dataset
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler


class IsoForestCICFlowExp(tune.Trainable):
    def _setup(self, cfg):
        # self.training_iteration = 0
        self.test_labels = None
        self.val_labels = None
        self.val_scores = None
        self.test_scores = None

        self.cfg = cfg

        trial_idx = cfg['__trial_index__']
        train, val = cfg['train_dates'][trial_idx]
        test = cfg['test_dates']

        self.dataset = CICFlowADDataset(root=os.path.abspath(cfg['data_path']),
                                        n_known_outlier_classes=1,
                                        train_dates=cfg['period'][train],
                                        val_dates=cfg['period'][val],
                                        test_dates=test,
                                        shuffle=True)

        def get_data_from_loader(loader):
            X = ()
            for data in loader:
                inputs, _, _, _ = data
                inputs = inputs.to(cfg['device'])
                X_batch = inputs.view(inputs.size(0), -1)
                X += (X_batch.cpu().data.numpy(), )
            return np.concatenate(X)

        self.isoforest = IsoForest(hybrid=False,
                                   n_estimators=int(cfg['n_estimators']),
                                   max_samples=cfg['max_samples'],
                                   contamination=cfg['contamination'],
                                   n_jobs=4,
                                   seed=cfg['seed'])

    def _train(self):
        # Train model on dataset
        self.isoforest.train(self.dataset,
                             device=self.cfg["device"],
                             n_jobs_dataloader=self.cfg["n_jobs_dataloader"])

        # Test model
        val_labels, val_scores = self.isoforest.test(
            self.dataset,
            device=self.cfg["device"],
            n_jobs_dataloader=self.cfg["n_jobs_dataloader"])
        test_labels, test_scores = self.isoforest.test(
            self.dataset,
            device=self.cfg["device"],
            n_jobs_dataloader=self.cfg["n_jobs_dataloader"])

        self.results = {
            "val": (val_labels, val_scores),
            "test": (test_labels, test_scores)
        }

        rocs = {
            phase + '_auc_roc': roc_auc_score(labels, scores)
            for phase in ["val", "test"]
            for labels, scores in [self.results[phase]]
        }

        prs = {
            phase + '_auc_pr': auc(recall, precision)
            for phase in ["val", "test"]
            for labels, scores in [self.results[phase]] for precision, recall, _ in
            [precision_recall_curve(labels, scores)]
        }

        return {**rocs, **prs}

    def _save(self, checkpoint_dir):
        pickle.dump(self.isoforest,
                    open(os.path.join(checkpoint_dir, 'isoforest.pkl'), "wb"))
        pickle.dump(self.results,
                    open(os.path.join(checkpoint_dir, 'results.pkl'), "wb"))
        return checkpoint_dir

    def _restore(self, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, 'isoforest.pkl'),
                  "wb") as pfile:
            return pickle.load(pfile)


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
    def _get_train_val_split(period, validation, n_splits=4):
        if (validation == 'kfold'):
            split = KFold(n_splits=n_splits)
        elif (validation == 'time_series'):
            split = TimeSeriesSplit(n_splits=n_splits)
        else:
            # Dummy object with split method that return indexes of train/test split 0.8/0.2. Similar to train_test_split without shuffle
            split = type(
                'obj', (object, ), {
                    'split':
                    lambda p: [([x for x in range(int(len(p) * 0.8))], [
                        x for x in range(int(len(p) * 0.8), len(p))
                    ])] * n_splits
                })

        return [(train, val) for train, val in split.split(period)]

    ray.init(address='auto')

    data_path = os.path.abspath(data_path)
    n_splits = 4

    # period = np.array([
    #     '2019-11-08', '2019-11-09', '2019-11-11', '2019-11-12', '2019-11-13',
    #     '2019-11-14', '2019-11-15'
    # ])
    period = np.array([
        '2019-11-08', '2019-11-09', '2019-11-11','2019-11-12'
    ])

    test_dates = period[-1:]
    train_dates = _get_train_val_split(period[:-2], validation, n_splits)

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
        name="IsoForestCICFlowExp",
        parameters=[
            {
                "name": "n_estimators",
                "type": "range",
                "bounds": [100, 1000],
            },
            {
                "name": "max_samples",
                "type": "range",
                "bounds": [1e-6, 1.0],
                "log_scale": True
            },
            {
                "name": "contamination",
                "type": "range",
                "bounds": [0.0, 0.15]
            },
        ],
        objective_name="val_auc_pr",
    )

    search_alg = AxSearch(ax)
    re_search_alg = Repeater(search_alg, repeat=n_splits)

    sched = ASHAScheduler(time_attr='training_iteration',
                          grace_period=10,
                          metric="val_auc_pr")

    analysis = tune.run(IsoForestCICFlowExp,
                        name="IsoForestCICFlowExp",
                        checkpoint_at_end=True,
                        checkpoint_freq=5,
                        stop={
                            "training_iteration": 1,
                        },
                        resources_per_trial={"cpu": 4},
                        num_samples=20,
                        local_dir=experiment_path,
                        search_alg=re_search_alg,
                        scheduler=sched,
                        config=exp_config)

    print("Best config is:", analysis.get_best_config(metric="val_auc_pr"))


if __name__ == '__main__':
    main()