from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

import numpy as np
import torch

import analysis
import utils
from active_learning import compute_utility_scores_entropy
from active_learning import compute_utility_scores_gap
from active_learning import compute_utility_scores_greedy
from model_extraction.deepfool import compute_utility_scores_deepfool
from datasets.utils import get_dataset_full_name
from datasets.utils import set_dataset
from datasets.utils import show_dataset_stats
from model_extraction.main_model_extraction import \
    run_model_extraction
from models.load_models import load_private_model_by_id
from models.load_models import load_private_models
from models.private_model import get_private_model_by_id
from models.utils_models import get_model_name_by_id
from models.utils_models import model_size
from parameters import get_parameters
from utils import from_result_to_str
from utils import get_unlabeled_indices
from utils import get_unlabeled_set
from utils import metric
from utils import result
from utils import train_model
from utils import update_summary


###########################
# ORIGINAL PRIVATE MODELS #
###########################
def train_private_models(args):
    """Train N = num-models private models."""
    start_time = time.time()

    # Checks
    assert 0 <= args.begin_id
    assert args.begin_id < args.end_id
    assert args.end_id <= args.num_models

    # Logs
    filename = 'logs-(id:{:d}-{:d})-(num-epochs:{:d}).txt'.format(
        args.begin_id + 1, args.end_id, args.num_epochs)
    if os.name == 'nt':
        filename = 'logs-(id_{:d}-{:d})-(num-epochs_{:d}).txt'.format(
            args.begin_id + 1, args.end_id, args.num_epochs)
    file = open(os.path.join(args.private_model_path, filename), 'w+')
    args.log_file = file
    args.save_model_path = args.private_model_path
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Training private models on '{}' dataset!".format(args.dataset), file)
    utils.augmented_print(
        "Training private models on '{}' architecture!".format(
            args.architecture), file)
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file)
    utils.augmented_print(f"Initial learning rate: {args.lr}.", file)
    utils.augmented_print(
        "Number of epochs for training each model: {:d}".format(
            args.num_epochs), file)

    # Data loaders

    all_private_trainloaders = utils.load_private_data(args=args)

    evalloader = utils.load_evaluation_dataloader(args)
    print(f'eval dataset: ', evalloader.dataset)

    if args.debug is True:
        # Logs about the eval set
        show_dataset_stats(dataset=evalloader.dataset, args=args, file=file,
                           dataset_name='eval')

    # Training
    summary = {
        'loss': [],
        'acc': [],
        'balanced_acc': [],
        'auc': [],
    }
    for id in range(args.begin_id, args.end_id):
        utils.augmented_print("##########################################",
                              file)

        # Private model for initial training.
        model = get_private_model_by_id(args=args, id=id)

        trainloader = all_private_trainloaders[id]

        print(f'train dataset for model id: {id}', trainloader.dataset)

        # Logs about the train set
        if args.debug is True:
            show_dataset_stats(dataset=trainloader.dataset,
                               args=args,
                               file=file,
                               dataset_name='private train')
        utils.augmented_print(
            "Steps per epoch: {:d}".format(len(trainloader)), file)

        train_model(
            args=args,
            model=model,
            trainloader=trainloader,
            evalloader=evalloader)
        result = eval_distributed_model(
            model=model, dataloader=evalloader, args=args)

        model_name = get_model_name_by_id(id=id)
        result['model_name'] = model_name
        result_str = from_result_to_str(result=result, sep=' | ',
                                        inner_sep=': ')
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)

        # Checkpoint
        state = result
        state['state_dict'] = model.state_dict()
        filename = "checkpoint-{}.pth.tar".format(model_name)
        filepath = os.path.join(args.private_model_path, filename)
        torch.save(state, filepath)

    utils.augmented_print("##########################################", file)

    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            utils.augmented_print(
                f"Average {key} of private models: {avg_value}", file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", file, flush=True)
    utils.augmented_print("##########################################", file)
    file.close()


def test_models(args):
    start_time = time.time()

    if args.num_querying_parties > 0:
        # Checks
        assert 0 <= args.begin_id
        assert args.begin_id < args.end_id
        assert args.end_id <= args.num_models
        args.querying_parties = range(args.begin_id, args.end_id, 1)
    else:
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        args.querying_parties = args.querying_party_ids

    # Logs
    filename = 'logs-testing-(id:{:d}-{:d})-(num-epochs:{:d}).txt'.format(
        args.begin_id + 1, args.end_id, args.num_epochs)
    file = open(os.path.join(args.private_model_path, filename), 'w')
    args.log_file = file

    test_type = args.test_models_type
    # test_type = 'retrained'
    # test_type = 'private'
    if test_type == 'private':
        args.save_model_path = args.private_model_path
    elif test_type == 'retrained':
        args.save_model_path = args.retrained_private_model_path
    else:
        raise Exception(f"Unknown test_type: {test_type}")

    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Test models on '{}' dataset!".format(args.dataset), file)
    utils.augmented_print(
        "Test models on '{}' architecture!".format(
            args.architecture), file)
    utils.augmented_print(
        "Number test models: {:d}".format(args.end_id - args.begin_id), file)

    evalloader = utils.load_evaluation_dataloader(args=args)
    # evalloader = utils.load_unlabeled_dataloader(args=args)
    # evalloader = utils.load_private_data(args=args)[0]
    print(f'eval dataset: ', evalloader.dataset)

    if args.debug is True:
        # Logs about the eval set
        show_dataset_stats(dataset=evalloader.dataset, args=args, file=file,
                           dataset_name='eval')

    # Training
    summary = {
        metric.loss: [],
        metric.acc: [],
        metric.balanced_acc: [],
        metric.auc: [],
        metric.map: [],
    }
    for id in args.querying_parties:
        utils.augmented_print("##########################################",
                              file)

        model = load_private_model_by_id(args=args, id=id,
                                         model_path=args.save_model_path)

        result = eval_distributed_model(
            model=model, dataloader=evalloader, args=args)

        model_name = get_model_name_by_id(id=id)
        result['model_name'] = model_name
        result_str = from_result_to_str(result=result, sep='\n',
                                        inner_sep=args.sep)
        utils.print_metrics_detailed(results=result)
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)

    utils.augmented_print("##########################################", file)

    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            std_value = np.std(value)
            min_value = np.min(value)
            max_value = np.max(value)
            med_value = np.median(value)
            str_value = utils.get_value_str(value=np.array(value))
            utils.augmented_print(
                f"{key} of private models;average;{avg_value};std;{std_value};"
                f"min;{min_value};max;{max_value};median;{med_value};"
                f"value;{str_value}", file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", file, flush=True)
    utils.augmented_print("##########################################", file)
    file.close()


def main(args):
    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # CUDA support
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_dataset(args=args)

    for model in args.architectures:
        args.architecture = model
        print('architecture: ', args.architecture)
        num_models_list = [args.num_models]
        for num_models in num_models_list:
            print('num_models: ', num_models)
            args.num_models = num_models
            if len(num_models_list) > 1:
                # for running experiments with many number of models
                args.end_id = num_models

            architecture = args.architecture
            dataset = get_dataset_full_name(args=args)
            # Folders
            args.private_model_path = os.path.join(
                args.path, 'private-models',
                dataset, architecture, '{:d}-models'.format(
                    args.num_models))
            print('args.private_model_path: ', args.private_model_path)
            args.save_model_path = args.private_model_path

            args.ensemble_model_path = os.path.join(
                args.path, 'ensemble-models',
                dataset, architecture, '{:d}-models'.format(
                    args.num_models))

            args.non_private_model_path = os.path.join(
                args.path, 'non-private-models',
                dataset, architecture)
            args.retrained_private_model_path = os.path.join(
                args.path,
                'retrained-private-models',
                dataset,
                architecture,
                '{:d}-models'.format(
                    args.num_models),
                args.mode)

            print('args.retrained_private_models_path: ',
                  args.retrained_private_model_path)
            addstr = ""
            if args.useserver:
                addstr += "pow"
            if args.target_model == "pate":
                addstr += "pate"
            if args.commands == ["adaptive_queries_only"]:
                addstr += "query"
            args.adaptive_model_path = os.path.join(
                args.path, 'adaptive-model',
                dataset, architecture, '{:d}-models'.format(
                    args.num_models), args.mode + addstr)
            if args.attacker_dataset:
                args.adaptive_model_path = os.path.join(
                    args.path, 'adaptive-model',
                    dataset + "_" + args.attacker_dataset, architecture,
                    '{:d}-models'.format(args.num_models), args.mode + addstr)

            for path_name in [
                'private_model',
                'ensemble_model',
                'retrained_private_model',
                'adaptive_model',
            ]:
                path_name += '_path'
                args_path = getattr(args, path_name)
                if os.path.exists(args_path):
                    raise Exception(
                        f'The {path_name}: {args_path} already exists.')
                else:
                    os.makedirs(args_path)
                # if not os.path.exists(args_path):
                #     os.makedirs(args_path)

            for command in args.commands:
                if command == 'train_private_models':
                    train_private_models(args=args)
                elif command in ["basic_model_stealing_attack", "basic_model_stealing_attack_with_BO"]:
                    run_model_extraction(args=args)
                elif command == "adaptive_queries_only":
                    run_model_extraction(args=args,no_model_extraction=True)
                else:
                    raise Exception(
                        'Unknown command: {}'.format(command))


if __name__ == '__main__':
    args = get_parameters()
    main(args)
