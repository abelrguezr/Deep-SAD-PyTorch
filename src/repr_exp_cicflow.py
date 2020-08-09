import click
import os
import pandas as pd
import torch
import logging
import random
import numpy as np
import logging
from datasets.nsl_kdd import NSLKDDADDataset
from datasets.cicflow import CICFlowADDataset
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from models.deepSVDD import DeepSVDD
from datasets.main import load_dataset

class CICFLOWExp():
    def _setup(self, params):
        # self.training_iteration = 0
        self.test_labels = None
        self.val_labels = None
        self.val_scores = None
        self.test_scores = None
        self.params = params
        self.cfg = params['cfg']

        # self.dataset = NSLKDDADDataset(root=os.path.abspath(params['data_path']),
        #                                n_known_outlier_classes=1,
        #                                shuffle=True)
        self.dataset = CICFlowADDataset(root=os.path.abspath(params['data_path']),
                                        n_known_outlier_classes=1,
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

        return self

    def _get_output(self):
        labels, outputs = self.model.get_output(self.dataset)

        c = self.model.trainer.c
        R = self.model.trainer.R
        

        print((c,R))
        
        results = locals().copy()
        del results["self"]

        with open('model_output.pkl','wb') as pfile:
            pickle.dump(results, pfile)


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
@click.option('--seed',
              type=int,
              default=0,
              help='Set seed. If -1, use randomization.')
def main(data_path, model_path, params_path, seed):


    data_path = os.path.abspath(data_path)
    params_path = os.path.abspath(params_path)
    model_path = os.path.abspath(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = pickle.load(open(params_path, "rb"))

    exp_config = {
        **locals().copy(),
        'objective': 'soft-boundary',
        'nu': 0.4
    }



    if exp_config['seed'] != -1:
        random.seed(exp_config['seed'])
        np.random.seed(exp_config['seed'])
        torch.manual_seed(exp_config['seed'])
        torch.cuda.manual_seed(exp_config['seed'])
        torch.backends.cudnn.deterministic = True

    d = CICFLOWExp()._setup(exp_config)
    d._get_output()

if __name__ == '__main__':
    main()