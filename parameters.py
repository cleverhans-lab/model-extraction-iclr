from getpass import getuser

import argparse
import os
from argparse import ArgumentParser

import utils
from models.utils_models import model_size
from models.utils_models import set_model_size
import numpy as np


def get_parameters():
    user = getuser()

    bool_params = []
    bool_choices = ['True', 'False']

    timestamp = utils.get_timestamp()

    commands = ['basic_model_stealing_attack']

    dataset = 'mnist'
    # dataset = 'fashion-mnist'
    # dataset = 'cifar10'
    # dataset = 'cifar100'
    # dataset = 'svhn'

    pick_labels = None
    num_querying_parties = 3
    adam_amsgrad = False
    dataset_type = 'balanced'
    optimizer = 'SGD'
    log_every_epoch = 0
    # debug = True
    debug = False
    if debug:
        num_workers = 0
    else:
        num_workers = 8
    begin_id = 0
    momentum = 0.9
    scheduler_type = 'ReduceLROnPlateau'
    scheduler_milestones = None
    loss_type = 'CE'

    default_model_size = None
    if num_workers > 0:
        device_ids = [0, 1, 2, 3]
    else:
        device_ids = [0]
        # device_ids = [1]

    querying_party_ids = [0, 1, 2]

    sigma_gnmax_private_knn = 28
    # selection_mode = 'random'
    selection_mode = 'entropy'

    if dataset == 'mnist':
        num_querying_parties = 3
        momentum = 0.5
        lr = 0.1
        weight_decay = 1e-4
        batch_size = 64
        eval_batch_size = 1000
        end_id = 1
        # num_epochs = 10
        num_epochs = 20
        num_models = 250

        selection_mode = 'random'
        # selection_mode = 'gap'
        # selection_mode = 'entropy'
        # selection_mode = 'deepfool'
        # selection_mode = 'greedy'
        # selection_mode = 'mixmatch'

        # threshold = 300
        # sigma_threshold = 200

        # threshold = 200
        # sigma_threshold = 150

        threshold = 0
        sigma_threshold = 0
        sigma_gnmax = 10.0
        sigma_gnmax_private_knn = 28
        budget = 10.0
        budgets = [budget]
        architecture = 'MnistNetPate'
        class_type = 'multiclass'
        dataset_type = 'balanced'
        default_model_size = model_size.small
        num_workers = 0
        device_ids = [0]
    elif dataset == 'cifar10':
        lr = 0.01
        weight_decay = 1e-5
        batch_size = 128
        eval_batch_size = batch_size
        end_id = 50
        num_epochs = 500
        num_models = 50
        threshold = 50.
        sigma_gnmax = 2.0
        sigma_gnmax_private_knn = 28
        sigma_threshold = 30.0
        budget = 20.0
        budgets = [budget]
        architecture = 'ResNet34'
        class_type = 'multiclass'
    elif dataset == 'svhn':
        lr = 0.1
        weight_decay = 1e-4
        batch_size = 128
        eval_batch_size = batch_size
        end_id = 1
        num_epochs = 200
        num_models = 250
        threshold = 0
        sigma_threshold = 0
        sigma_gnmax = 10.0
        sigma_gnmax_private_knn = 28
        budget = 3.0
        budgets = [budget]
        architecture = 'ResNet34'
        if architecture.startswith('ResNet'):
            lr = 0.01
            weight_decay = 1e-5
            num_epochs = 300
        class_type = 'multiclass'
    else:
        raise Exception('Unknown dataset: {}'.format(dataset))

    if debug is True:
        debug = 'True'
    else:
        debug = 'False'

    parser = argparse.ArgumentParser(
        description='Model Extraction Proof of Work')
    parser.add_argument('--timestamp',
                        type=str,
                        default=timestamp,
                        help='timestamp')
    parser.add_argument('--path', type=str,
                        default=f'/home/{user}/code/model-extraction-pow',
                        help='path to the project')
    parser.add_argument('--data_dir', type=str,
                        default=f'/home/{user}/data',
                        help='path to the data')
    # General parameters
    parser.add_argument('--dataset', type=str,
                        default=dataset,
                        help='name of the dataset')
    parser.add_argument(
        '--class_type',
        type=str,
        default=class_type,
        help='The type of the classification: binary, multiclass with a '
             'single class per data item, and multilabel classification with '
             'zero or more possible classes assigned to a data item.',
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default=dataset_type,
        # default='balanced',
        # default='imbalanced',
        help='Type of the dataset.')
    parser.add_argument('--begin_id', type=int, default=begin_id,
                        help='train private models with id number in [begin_id, end_id)')
    parser.add_argument('--end_id', type=int, default=end_id,
                        help='train private models with id number in [begin_id, end_id)')
    parser.add_argument('--num_querying_parties', type=int,
                        default=num_querying_parties,
                        help='number of parties that pose queries')
    parser.add_argument('--querying_party_ids', type=int, nargs='+',
                        default=querying_party_ids,
                        help='the id of the querying party')
    parser.add_argument('--mode',
                        type=str,
                        # default='random',
                        # default='entropy',
                        # default='gap',
                        # default='greedy',
                        # default='deepfool',
                        default=selection_mode,
                        help='method for generating utility scores')
    parser.add_argument('--verbose',
                        default='True',
                        # default=False,
                        type=str,
                        choices=bool_choices,
                        help="Detail info")
    bool_params.append('verbose')
    parser.add_argument('--debug',
                        default=debug,
                        # default=False,
                        type=str,
                        choices=bool_choices,
                        help="Debug mode of execution")
    bool_params.append('debug')
    parser.add_argument('--sep',
                        default=';',
                        type=str,
                        help="Separator for the output log.")
    parser.add_argument('--log_every_epoch',
                        default=log_every_epoch,
                        type=int,
                        help="Log test accuracy every n epchos.")

    # Training parameters
    parser.add_argument('--optimizer', type=str,
                        default=optimizer,
                        # default='SGD',
                        help='The type of the optimizer.')
    parser.add_argument('--adam_amsgrad', type=bool,
                        default=adam_amsgrad,
                        help='amsgrad param for Adam optimizer')
    parser.add_argument('--loss_type',
                        type=str,
                        default=loss_type,
                        # default='CE',
                        help='The type of the loss (e.g., MSE, CE, BCE, etc.).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=batch_size,
                        help='batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=eval_batch_size,
                        help='batch size for evaluation')
    parser.add_argument('--adaptive_batch_size', type=int, default=250,
                        help='batch size for adaptive training')
    parser.add_argument('--patience', type=int, default=None,
                        help='patience for adaptive training')
    parser.add_argument("--target_model", type=str,
                        default="pate",
                        help='pate when a pate ensemble is used for privacy calculation, victim for a single model behind API')
    parser.add_argument(
        '--shuffle_dataset', action='store_true',
        default=False,
        help='shuffle dataset before split to train private models.  '
             'only implemented for mnist')
    parser.add_argument(
        '--num_optimization_loop', type=int,
        default=0,
        help='num_optimization_loop for adaptive training with bayesian '
             'optimization')

    parser.add_argument('--momentum', type=float, default=momentum,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=weight_decay,
                        help='L2 weight decay factor')
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--lr', type=float, default=lr,
                        help='initial learning rate')
    parser.add_argument('--lr_factor', type=float,
                        default=0.1,
                        help='learning rate decay factor')
    parser.add_argument('--lr_epochs',
                        type=int,
                        nargs='+',
                        default=[2],
                        help='Epoch when learning rate decay occurs.')
    parser.add_argument('--lr_mixmatch', type=float, default=0.002,
                        # value from pytorch implementation is 0.002 (paper as well)
                        help='initial learning rate for mix match')
    parser.add_argument('--num_epochs', type=int, default=num_epochs,
                        help='number of epochs for training')
    parser.add_argument(
        "--attacker_dataset",
        default=None,
        type=str,
        help="dataset used by model extraction attack, default to be the same as dataset")
    parser.add_argument(
        '--architectures',
        nargs='+',
        type=str,
        default=[architecture],
        help='The architectures of heterogeneous models.',
    )
    parser.add_argument(
        '--model_size',
        type=model_size,
        choices=list(model_size),
        default=default_model_size,
        help='The size of the model.'
    )
    parser.add_argument(
        '--device_ids',
        nargs='+',
        type=int,
        default=device_ids,
        help='Cuda visible devices.')
    parser.add_argument(
        '--scheduler_type',
        type=str,
        default=scheduler_type,
        # default='ReduceLROnPlateau',
        # default='MultiStepLR',
        help='Type of the scheduler.')
    parser.add_argument(
        '--scheduler_milestones',
        nargs='+',
        type=int,
        default=scheduler_milestones,
        help='The milestones for the multi-step scheduler.'
    )
    parser.add_argument(
        '--schedule_factor',
        type=float,
        default=0.1,
        help='The factor for scheduler.'
    )
    parser.add_argument(
        '--schedule_patience',
        type=int,
        default=10,
        help='The patience for scheduler.'
    )
    parser.add_argument('--num_workers', type=int, default=num_workers,
                        help='Number of workers to fetch data.')

    # Privacy parameters
    parser.add_argument('--num_models', type=int, default=num_models,
                        help='number of private models')
    parser.add_argument('--threshold', type=float, default=threshold,
                        help='threshold value (a scalar) in the threshold mechanism')
    parser.add_argument('--sigma_gnmax', type=float,
                        default=sigma_gnmax,
                        help='std of the Gaussian noise in the GNMax mechanism')
    parser.add_argument('--sigma_gnmax_private_knn', type=float,
                        default=sigma_gnmax_private_knn,
                        help='std of the Gaussian noise in the GNMax mechanism used for the pknn cost')
    parser.add_argument('--sigma_threshold', type=float,
                        default=sigma_threshold,
                        help='std of the Gaussian noise in the threshold mechanism')
    parser.add_argument('--budget', type=float, default=budget,
                        help='pre-defined epsilon value for (eps, delta)-DP')
    parser.add_argument('--budgets', nargs="+",
                        type=float, default=budgets,
                        help='pre-defined epsilon value for (eps, delta)-DP')

    parser.add_argument(
        '--poisson_mechanism',
        default='False',
        type=str,
        choices=bool_choices,
        help="Apply or disable the poisson mechanism.")
    bool_params.append('poisson_mechanism')

    # Command parameters (what to run).
    parser.add_argument(
        '--commands',
        nargs='+',
        type=str,
        default=commands,
        help='which commands to run')

    parser.add_argument(
        '--query_set_type', type=str,
        default='raw',
        # default='numpy',
        help='The type of query set saved for the retraining when we query the'
             'ensemble of the teacher models.'
    )

    parser.add_argument(
        '--test_models_type', type=str,
        # default='retrained',
        default='private',
        help='The type of models to be tested.'
    )

    parser.add_argument(
        '--transfer_type', type=str,
        # default='cross-domain',
        default='',
        help='The transfer of knowledge can be cross-domain, e.g., from the '
             'chexpert ensemble to the padchest models.'
    )

    parser.add_argument(
        '--sigmoid_op', type=str,
        default='apply',
        # default='disable',
        help='Apply or disable the sigmoid operation outside of model arhictecture.'
    )

    parser.add_argument(
        '--label_reweight', type=str,
        # default='apply',
        default='disable',
        help='Apply or disable the label reweighting based on the balanced '
             'accuracy found on the privately trained model.'
    )

    parser.add_argument(
        '--show_dp_budget', type=str,
        default='apply',
        # default='disable',
        help='Apply or disable showing the current privacy budget.'
    )

    parser.add_argument('--apply_data_independent_bound',
                        default='False',
                        type=str,
                        choices=bool_choices,
                        help="Disable it in case of the privacy estimate for "
                             "model extraction.")
    bool_params.append('apply_data_independent_bound')

    parser.add_argument('--retrain_extracted_model',
                        default='False',
                        type=str,
                        choices=bool_choices,
                        help="Do we re-train the extracted / stolen model on the newly labeled data?")
    bool_params.append('retrain_extracted_model')

    parser.add_argument('--useserver',
                        default='False',
                        type=str,
                        choices=bool_choices,
                        help="Use the server client setup with PoW for querying")
    bool_params.append('useserver')

    args = parser.parse_args()
    args.cwd = os.getcwd()

    for param in bool_params:
        transform_bool(args=args, param=param)

    # os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.device_ids}'
    print_args(args=args)

    set_model_size(args=args)

    return args


def transform_bool(args, param: str):
    """
    Transform the string boolean params to python bool values.

    :param args: program args
    :param param: name of the boolean param
    """
    attr_value = getattr(args, param, None)
    if attr_value is None:
        raise Exception(f"Unknown param in args: {param}")
    if attr_value == 'True':
        setattr(args, param, True)
    elif attr_value == 'False':
        setattr(args, param, False)
    else:
        raise Exception(
            f"Unknown value for the args.{param}: {attr_value}.")


def print_args(args, get_str=False):
    if 'delimiter' in args:
        delimiter = args.delimiter
    elif 'sep' in args:
        delimiter = args.sep
    else:
        delimiter = ';'
    print('###################################################################')
    print('args: ')
    keys = sorted(
        [a for a in dir(args) if not (
                a.startswith('__') or a.startswith(
            '_') or a == 'sep' or a == 'delimiter')])
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        print(keys_str)
        print(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            print(key, ': ', value, flush=True)
    print('ARGS FINISHED', flush=True)
    print('######################################################')
