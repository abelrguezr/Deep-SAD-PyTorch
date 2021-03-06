import click
import os
import torch
import logging
import random
import numpy as np
import ray
import pickle
from utils.misc import get_ratio_anomalies
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from ray import tune
from ray.tune import track
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
from sklearn.model_selection import TimeSeriesSplit, KFold, train_test_split
from datasets.cicflow import CICFlowADDataset
from models.deepSVDD import DeepSVDD
from datasets.main import load_dataset
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler


class OneDaySVDDCICFlowExp(tune.Trainable):
    def _setup(self, cfg):
        # self.training_iteration = 0
        self.test_labels = None
        self.val_labels = None
        self.val_scores = None
        self.test_scores = None

        dates = cfg['dates']

        self.dataset = CICFlowADDataset(root=os.path.abspath(cfg['data_path']),
                                        n_known_outlier_classes=1,
                                        test_dates=dates,
                                        shuffle=True,
                                        split=True)

        self.model = DeepSVDD(cfg['objective'], cfg['nu'])
        self.model.set_trainer(optimizer_name=cfg['optimizer_name'],
                               lr=cfg['lr'],
                               n_epochs=cfg['n_epochs'],
                               lr_milestones=cfg['lr_milestone'],
                               batch_size=cfg['batch_size'],
                               weight_decay=cfg['weight_decay'],
                               device=cfg['device'],
                               n_jobs_dataloader=cfg["n_jobs_dataloader"])
        self.model.setup(self.dataset, cfg['net_name'])

        if cfg['pretrain']:
            self.model = self.model.pretrain(
                self.dataset,
                optimizer_name=cfg['optimizer_name'],
                lr=cfg['lr'],
                n_epochs=cfg['ae_n_epochs'],
                lr_milestones=cfg['ae_lr_milestone'],
                batch_size=cfg['ae_batch_size'],
                weight_decay=cfg['ae_weight_decay'],
                device=cfg['device'],
                n_jobs_dataloader=cfg["n_jobs_dataloader"])

    def _train(self):
        self.model.train_one_step(self.training_iteration)
        self.model.test(self.dataset, set_split="train")
        self.model.test(self.dataset, set_split="val")
        self.model.test(self.dataset, set_split="test")

        val_labels, val_scores, _ = self.model.trainer.get_results("val")
        test_labels, test_scores, _ = self.model.trainer.get_results("test")
        train_labels, train_scores, _ = self.model.trainer.get_results("train")

        results = locals().copy()
        del results["self"]

        self.results = results

        rocs = {
            phase + '_auc_roc': roc_auc_score(labels, scores)
            for phase in ["val", "test", "train"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
        }
        
        ratios = {
            phase + '_ratio_anomalies': get_ratio_anomalies(labels)
            for phase in ["val", "test", "train"]
            for labels, _, _ in [self.model.trainer.get_results(phase)]
        }

        prc = {
            phase + '_auc_pr': auc(recall, precision)
            for phase in ["val", "test", "train"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
            for precision, recall, _ in
            [precision_recall_curve(labels, scores)]
        }

        get_f1 = lambda pr, rec: 2 * (pr * rec) / (pr + rec)
        max_f1 = lambda pr, rec: max(get_f1(pr, rec))
        idx_max_f1 = lambda pr, rec: np.argmax(
            get_f1(pr, rec)[~np.isnan(get_f1(pr, rec))])

        f1s = {
            phase + '_max_f1': max_f1(precision, recall)
            for phase in ["val", "test", "train"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
            for precision, recall, _ in
            [precision_recall_curve(labels, scores)]
        }
        prs = {
            phase + '_precision_max_f1':
            precision[idx_max_f1(precision, recall)]
            for phase in ["val", "test", "train"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
            for precision, recall, _ in
            [precision_recall_curve(labels, scores)]
        }

        recs = {
            phase + '_recall_max_f1': recall[idx_max_f1(precision, recall)]
            for phase in ["val", "test", "train"]
            for labels, scores, _ in [self.model.trainer.get_results(phase)]
            for precision, recall, _ in
            [precision_recall_curve(labels, scores)]
        }

        return {**rocs, **ratios, **prc, **prs, **recs, **f1s}


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

    dates = period[:2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_config = {
        **locals().copy(),
        'objective': 'soft-boundary'

    }

    if exp_config['seed'] != -1:
        random.seed(exp_config['seed'])
        np.random.seed(exp_config['seed'])
        torch.manual_seed(exp_config['seed'])
        torch.cuda.manual_seed(exp_config['seed'])
        torch.backends.cudnn.deterministic = True

    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(
        name="SVDDCICFlowExp",
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-6, 0.1],
                "log_scale": True
            },
            {
                "name": "nu",
                "type": "range",
                "bounds": [0.005, 0.5]
            },
            {
                "name": "net_name",
                "type": "choice",
                "values": ['cicflow_mlp','cicflow_mlp_2','cicflow_mlp_3']
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [1e-6, 0.01],
                "log_scale": True
            },
        ],
        objective_name="val_auc_pr",
    )

    search_alg = AxSearch(ax)

    analysis = tune.run(OneDaySVDDCICFlowExp,
                        name="OneDaySVDDCICFlowExp",
                        checkpoint_at_end=True,
                        checkpoint_freq=5,
                        stop={
                            "training_iteration": 50,
                        },
                        resources_per_trial={"gpu": 1},
                        num_samples=10,
                        local_dir=experiment_path,
                        search_alg=search_alg,
                        config=exp_config)

    print("Best config is:", analysis.get_best_config(metric="val_auc_pr"))


if __name__ == '__main__':
    main()