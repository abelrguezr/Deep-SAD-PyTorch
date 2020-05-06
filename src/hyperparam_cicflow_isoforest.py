import click
import os
import torch
import logging
from filelock import FileLock
import random
import numpy as np
import logging
import ray
from ray import tune
from ray.tune import track
from ray.tune.suggest.ax import AxSearch
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient
from sklearn.model_selection import TimeSeriesSplit, KFold, train_test_split
from datasets.cicflow import CICFlowADDataset
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from baselines.isoforest import IsoForest
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name',
                type=click.Choice([
                    'mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio',
                    'satellite', 'satimage-2', 'shuttle', 'thyroid', 'cicflow'
                ]))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config',
              type=click.Path(exists=True),
              default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model',
              type=click.Path(exists=True),
              default=None,
              help='Model file path (default: None).')
@click.option('--ratio_known_normal',
              type=float,
              default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier',
              type=float,
              default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option(
    '--ratio_pollution',
    type=float,
    default=0.0,
    help=
    'Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.'
)
@click.option('--seed',
              type=int,
              default=0,
              help='Set seed. If -1, use randomization.')
@click.option('--validation',
              type=click.Choice(['kfold', 'time_series', 'index']),
              default='kfold',
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
def main(dataset_name, xp_path, data_path, load_config, load_model,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, validation,
         n_jobs_dataloader, normal_class, known_outlier_class, seed,
         n_known_outlier_classes):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    ######################################################
    #                  GLOBAL CONFIG                     #
    ######################################################

    xp_path = os.path.abspath(xp_path)
    data_path = os.path.abspath(data_path)
    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(tune.__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' %
                ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' %
                ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' %
                ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' %
                    n_known_outlier_classes)

    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    ######################################################
    #                       EXP CONFIG                   #
    ######################################################

    # Init ray
    ray.init(address='auto')
    ax = AxClient(enforce_sequential_optimization=False)
    # Default device to 'cpu' if cuda is not available

    ax.create_experiment(
        name="cicflow_mlp_experiment",
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
        objective_name="mean_auc",
    )

    def isoforest_trainable(parameterization):
        return train_evaluate(parameterization,
                              validation=validation,
                              data_path=data_path,
                              n_known_outlier_classes=n_known_outlier_classes,
                              ratio_known_normal=ratio_known_normal,
                              ratio_known_outlier=ratio_known_outlier,
                              cfg=cfg,
                              n_jobs_dataloader=n_jobs_dataloader,
                              ratio_pollution=ratio_pollution)

    tune.run(
        isoforest_trainable,
        num_samples=30,
        resources_per_trial={'cpu': 4},
        search_alg=AxSearch(
            ax),  # Note that the argument here is the `AxClient`.
        verbose=
        2,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        # To use GPU, specify: resources_per_trial={"gpu": 1}.
    )

    best_parameters, values = ax.get_best_parameters()
    best_parameters


def train_evaluate(parameterization,
                   validation,
                   data_path,
                   n_known_outlier_classes,
                   ratio_known_normal,
                   ratio_known_outlier,
                   ratio_pollution,
                   cfg,
                   n_jobs_dataloader,
                   n_splits=3):

    device = 'cpu'

    period = np.array(['2019-11-08','2019-11-09','2019-11-11','2019-11-12','2019-11-13'])

    if (validation == 'kfold'):
        split = KFold(n_splits=n_splits)
    elif (validation == 'time_series'):
        split = TimeSeriesSplit(n_splits=n_splits)
    else:
        # Dummy object with split method that return indexes of train/test split 0.8/0.2. Similar to train_test_split without shuffle
        split = type(
            'obj', (object, ), {
                'split':
                lambda p: [([x for x in range(int(len(p) * 0.8))],
                            [x for x in range(int(len(p) * 0.8), len(p))])]
            })

    test_aucs = []

    for train, test in split.split(period):

        dataset = CICFlowADDataset(
            root=os.path.abspath(data_path),
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            train_dates=period[train],
            test_dates=period[test],
            ratio_pollution=ratio_pollution)

        # Initialize DeepSAD model and set neural network phi

        # Log random sample of known anomaly classes if more than 1 class
        if n_known_outlier_classes > 1:
            logger.info('Known anomaly classes: %s' %
                        (dataset.known_outlier_classes, ))

        # Initialize Isolation Forest model
        Isoforest = IsoForest(hybrid=False,
                              n_estimators=int(
                                  parameterization['n_estimators']),
                              max_samples=parameterization['max_samples'],
                              contamination=parameterization['contamination'],
                              n_jobs=4,
                              seed=cfg.settings['seed'])

        # Train model on dataset
        Isoforest.train(dataset,
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader)

        # Test model
        Isoforest.test(dataset,
                       device=device,
                       n_jobs_dataloader=n_jobs_dataloader)

        test_auc = Isoforest.results['auc_roc']

        test_aucs.append(test_auc)

    track.log(mean_auc=evaluate_aucs(test_aucs=test_aucs))


def evaluate_aucs(test_aucs):
    return sum(test_aucs) / len(test_aucs)


if __name__ == '__main__':
    main()