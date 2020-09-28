import click
import os
import pandas as pd
import torch
import logging
import random
import numpy as np
import logging
import ray
from itertools import tee
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from ray import tune
from ray.tune import track
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
from sklearn.model_selection import TimeSeriesSplit, KFold, train_test_split
from datasets.cicflow import CICFlowADDataset
from networks.mlp import MLP
from models.deepSVDD import DeepSVDD
from datasets.main import load_dataset
from ray.tune.suggest import Repeater


class DriftCICFlowExp(tune.Trainable):
    def _setup(self, params):
        # self.training_iteration = 0
        self.test_labels = None
        self.val_labels = None
        self.val_scores = None
        self.test_scores = None
        self.params = params
        self.cfg = params['cfg']
        self.incremental = params['incremental']
        self.dates = self._get_train_test(params['dates'])

        self.dataset = CICFlowADDataset(root=os.path.abspath(
            self.params['data_path']),
                                        n_known_outlier_classes=1,
                                        train_dates=[params['dates'][0]],
                                        val_dates=[params['dates'][0]],
                                        test_dates=[params['dates'][0]],
                                        shuffle=True)

        self.model = DeepSVDD(self.cfg['objective'], self.cfg['nu'])
        self.model.set_trainer(optimizer_name=self.cfg['optimizer_name'],
                               lr=self.cfg['lr'],
                               n_epochs=self.cfg['n_epochs'],
                               lr_milestones=self.cfg['lr_milestone'],
                               batch_size=self.cfg['batch_size'],
                               weight_decay=self.cfg['weight_decay'],
                               device=self.params['device'],
                               n_jobs_dataloader=self.cfg["n_jobs_dataloader"])
        self.model.setup(self.dataset, self.cfg['net_name'])
        self.model.load_model(params['model_path'])
        self.model.test(self.dataset)

    def _get_train_test(self, dates):
        train, test = tee(dates)
        next(test, None)
        return zip(train, test)

    def _train(self):
        try:
            train, test = next(self.dates)
        except StopIteration:
            return {'done': True}

        self.dataset = CICFlowADDataset(root=os.path.abspath(
            self.params['data_path']),
                                        n_known_outlier_classes=1,
                                        train_dates=[train],
                                        val_dates=[train],
                                        test_dates=[test],
                                        shuffle=True)

        if self.incremental:
            self.model.train(dataset=self.dataset,
                       optimizer_name=self.cfg['optimizer_name'],
                       lr=self.cfg['lr'],
                       n_epochs=1,
                       lr_milestones=self.cfg['lr_milestone'],
                       batch_size=self.cfg['batch_size'],
                       weight_decay=self.cfg['weight_decay'],
                       device=self.params['device'],
                       n_jobs_dataloader=self.cfg["n_jobs_dataloader"])

        self.model.test(self.dataset, set_split="test")
        self.model.test(self.dataset, set_split="train")

        test_labels, test_scores, _ = self.model.trainer.get_results("test")

        results = locals().copy()
        del results["self"]

        self.results = results

        rocs = {
            phase + '_auc_roc': roc_auc_score(labels, scores)
            for phase in ["test"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
        }

        prs = {
            phase + '_auc_pr': auc(recall, precision)
            for phase in ["test"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
            for precision, recall, _ in
            [precision_recall_curve(labels, scores)]
        }

        return {**rocs, **prs}

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
@click.option('--model_path',
              type=click.Path(exists=True),
              default=None,
              help='Model file path (default: None).')
@click.option('--params_path',
              type=click.Path(exists=True),
              default=None,
              help='Model file path (default: None).')
@click.option('--experiment_path',
              type=click.Path(exists=True),
              default='~/ray_results',
              help='Model file path (default: None).')
@click.option('--seed',
              type=int,
              default=0,
              help='Set seed. If -1, use randomization.')
def main(data_path, experiment_path, model_path, params_path, seed):

    ray.init(address='auto')

    data_path = os.path.abspath(data_path)
    params_path = os.path.abspath(params_path)
    experiment_path = os.path.abspath(experiment_path)
    model_path = os.path.abspath(model_path)
    n_splits = 4

    dates = np.array(['2019-11-09', '2019-11-11'])
    # period = np.array([
    #     '2019-11-08', '2019-11-09', '2019-11-11', '2019-11-12', '2019-11-13',
    #     '2019-11-14', '2019-11-15'
    # ])

    # dates = period[:7]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = pickle.load(open(params_path, "rb"))

    exp_config = {
        **locals().copy(),
        "incremental": tune.grid_search([True, False])
    }

    if exp_config['seed'] != -1:
        random.seed(exp_config['seed'])
        np.random.seed(exp_config['seed'])
        torch.manual_seed(exp_config['seed'])
        torch.cuda.manual_seed(exp_config['seed'])
        torch.backends.cudnn.deterministic = True

    analysis = tune.run(DriftCICFlowExp,
                        name="DriftCICFlowExp",
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        stop={
                            "training_iteration": len(dates),
                        },
                        resources_per_trial={"gpu": 0},
                        num_samples=1,
                        local_dir=experiment_path,
                        config=exp_config)


if __name__ == '__main__':
    main()