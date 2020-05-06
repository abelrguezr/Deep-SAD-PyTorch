import click
import os
import torch
import logging
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
from DeepSAD import DeepSAD
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
@click.argument('net_name',
                type=click.Choice([
                    'mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet',
                    'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp',
                    'satimage-2_mlp', 'shuttle_mlp', 'cicflow_mlp',
                    'cicflow_tcn', 'thyroid_mlp'
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
@click.option('--eta',
              type=float,
              default=1.0,
              help='Deep SAD hyperparameter eta (must be 0 < eta).')
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
@click.option('--device',
              type=str,
              default='cuda',
              help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).'
              )
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
              default='index',
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
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model,
         eta, ratio_known_normal, ratio_known_outlier, ratio_pollution, device,
         seed, optimizer_name, validation, lr, n_epochs, lr_milestone,
         batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class,
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
    logger.info('Network: %s' % net_name)

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
                "name": "lr",
                "type": "range",
                "bounds": [1e-6, 0.4],
                "log_scale": True
            },
            {
                "name": "eta",
                "type": "range",
                "bounds": [0.0, 1.5]
            },
        ],
        objective_name="mean_auc",
    )

    def mlp_trainable(parameterization, reporter):
        return train_evaluate(parameterization,
                              reporter,
                              validation=validation,
                              data_path=data_path,
                              n_known_outlier_classes=n_known_outlier_classes,
                              ratio_known_normal=ratio_known_normal,
                              ratio_known_outlier=ratio_known_outlier,
                              cfg=cfg,
                              n_jobs_dataloader=n_jobs_dataloader,
                              net_name=net_name,
                              pretrain=pretrain,
                              ratio_pollution=ratio_pollution)

    tune.run(
        mlp_trainable,
        name = "Hyperparam MLP"
        num_samples=30,
        resources_per_trial={'gpu': 1},
        search_alg=AxSearch(
            ax),  # Note that the argument here is the `AxClient`.
        verbose=
        2,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        # To use GPU, specify: resources_per_trial={"gpu": 1}.
    )

    best_parameters, values = ax.get_best_parameters()
    best_parameters


def train_evaluate(parameterization,
                   reporter,
                   validation,
                   data_path,
                   n_known_outlier_classes,
                   ratio_known_normal,
                   ratio_known_outlier,
                   ratio_pollution,
                   cfg,
                   n_jobs_dataloader,
                   net_name,
                   pretrain,
                   n_splits=5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    period = np.array(
        ['2019-11-08', '2019-11-09', '2019-11-11', '2019-11-12', '2019-11-13'])
    # period = np.array(['2019-11-08','2019-11-09'])

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

    for train, test in (split.split(period)):

        dataset = CICFlowADDataset(
            root=os.path.abspath(data_path),
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            train_dates=period[train],
            test_dates=period[test],
            ratio_pollution=ratio_pollution,
            shuffle=True)

        # Initialize DeepSAD model and set neural network phi

        model = DeepSAD(parameterization['eta']).set_network(net_name)

        if pretrain:

            model = model.pretrain(
                dataset,
                optimizer_name=cfg.settings['ae_optimizer_name'],
                lr=parameterization['lr'],
                n_epochs=cfg.settings['ae_n_epochs'],
                lr_milestones=cfg.settings['ae_lr_milestone'],
                batch_size=cfg.settings['ae_batch_size'],
                weight_decay=cfg.settings['ae_weight_decay'],
                device=device,
                n_jobs_dataloader=n_jobs_dataloader)

            # Save pretraining results
            # deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

        # Train model on dataset

        model = model.train(dataset,
                            optimizer_name=cfg.settings['ae_optimizer_name'],
                            lr=parameterization['lr'],
                            n_epochs=cfg.settings['n_epochs'],
                            lr_milestones=cfg.settings['lr_milestone'],
                            batch_size=cfg.settings['batch_size'],
                            weight_decay=cfg.settings['weight_decay'],
                            device=device,
                            n_jobs_dataloader=n_jobs_dataloader,
                            reporter=reporter)

        model.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
        test_auc = model.results['auc_roc']

        test_aucs.append(test_auc)

    track.log(mean_auc=evaluate_aucs(test_aucs=test_aucs))


def evaluate_aucs(test_aucs):
    return sum(test_aucs) / len(test_aucs)


if __name__ == '__main__':
    main()
