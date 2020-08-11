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

        dates = np.array([cfg['dates']])

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
        self.model.load_model(cfg['model_path'])


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
@click.option('--experiment_path',
              type=click.Path(exists=True),
              default='~/ray_results',
              help='Model file path (default: None).')
@click.option('--model_path',
              type=click.Path(exists=True),
              default=None,
              help='Model file path (default: None).')
@click.option('--params_path',
              type=click.Path(exists=True),
              default=None,
              help='Model file path (default: None).')
def main(data_path, experiment_path, model_path, params_path):
    ray.init(address='auto')

    data_path = os.path.abspath(data_path)
    params_path = os.path.abspath(params_path)
    model_path = os.path.abspath(model_path)    
    n_splits = 4

    cfg = pickle.load(open(params_path, "rb"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_config = {
        **locals().copy(),
        **cfg,
        'objective': 'soft-boundary',
        'net_name':'cicflow_mlp_2',

    }

    if exp_config['seed'] != -1:
        random.seed(exp_config['seed'])
        np.random.seed(exp_config['seed'])
        torch.manual_seed(exp_config['seed'])
        torch.cuda.manual_seed(exp_config['seed'])
        torch.backends.cudnn.deterministic = True

    dates = ['2019-11-08', '2019-11-09', '2019-11-11', '2019-11-12', '2019-11-13',
        '2019-11-14', '2019-11-15','2019-11-16','2019-11-17','2019-11-18','2019-11-19']

    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(
        name="SVDDCICFlowExp",
        parameters=[
            {
                "name": "dates",
                "type": "choice",
                "values": dates
            },
        ],
        objective_name="val_auc_pr",
    )

    search_alg = AxSearch(ax)

    analysis = tune.run(OneDaySVDDCICFlowExp,
                        name="DriftSVDDCICFlowExp",
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        stop={
                            "training_iteration": 1,
                        },
                        resources_per_trial={"gpu": 1},
                        num_samples=len(dates),
                        local_dir=experiment_path,
                        search_alg=search_alg,
                        config=exp_config)

    print("Best config is:", analysis.get_best_config(metric="val_auc_pr"))


if __name__ == '__main__':
    main()