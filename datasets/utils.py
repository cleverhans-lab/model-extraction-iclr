import getpass
import json
import numpy as np
import warnings

import os

from utils import get_cfg, class_ratio, augmented_print

def set_dataset(args):
    # Dataset
    args.dataset = args.dataset.lower()
    args.datasets = ['mnist', 'fashion-mnist', 'svhn', 'cifar10', 'cifar100']
    args.datasets_string = ",".join(args.datasets)
    args.datasets_exception = \
        f'Dataset name must be in: {args.datasets_string}. Check if the ' \
        f'option is supported for your dataset.'
    user = getpass.getuser()
    if args.dataset == 'mnist':
        args.dataset_path = os.path.join(args.data_dir, 'MNIST')
        args.num_unlabeled_samples = 9000
        args.num_dev_samples = 0
        args.num_classes = 10
        # Hyper-parameter delta in (eps, delta)-DP.
        args.delta = 1e-5
        args.num_teachers_private_knn = 300
        #args.sigma_gnmax_private_knn = 28
    elif args.dataset == 'fashion-mnist':
        args.dataset_path = os.path.join(args.data_dir, 'Fashion-MNIST')
        args.num_unlabeled_samples = 9000
        args.num_classes = 10
        args.delta = 1e-5
        args.num_teachers_private_knn = 300
        #args.sigma_gnmax_private_knn = 28
    elif args.dataset == 'svhn':
        args.dataset_path = os.path.join(args.data_dir, 'SVHN')
        args.num_unlabeled_samples = 25000
        args.num_classes = 10
        args.delta = 1e-6
        args.num_teachers_private_knn = 300
        #rgs.sigma_gnmax_private_knn = 28
    elif args.dataset == 'imagenet': 
        args.dataset_path = os.path.join(args.data_dir, 'imagenet')
        args.num_unlabeled_samples = 50000
        args.num_classes = 1000
        args.delta = 1e-8
        args.num_teachers_private_knn = 800
        #args.sigma_gnmax_private_knn = 2
    elif args.dataset == 'cifar10':
        args.dataset_path = os.path.join(args.data_dir, 'CIFAR10')
        args.num_unlabeled_samples = 9000
        args.num_classes = 10
        args.delta = 1e-5
        args.num_teachers_private_knn = 300
        args.sigma_gnmax_private_knn = 28
    elif args.dataset == 'cifar100':
        args.dataset_path = os.path.join(args.data_dir, 'CIFAR100')
        args.num_unlabeled_samples = 9000
        args.num_classes = 100
        args.delta = 1e-5
    else:
        raise Exception(
            f"For dataset: {args.dataset}. " + args.datasets_exception)


def get_dataset_full_name(args):
    dataset = args.dataset

    return dataset


def show_dataset_stats(dataset, file, args, dataset_name=''):
    """
    Show statistics about this dataset.

    :param dataset: the loader for the dataset
    :param file: where to write the log
    :param args: arguments
    :param dataset_name: is it test or train
    :return: nothing
    """
    counts, ratios = class_ratio(dataset, args)
    label_counts = np.array2string(counts, separator=', ')
    augmented_print(
        f"Label counts for {dataset_name} set: {label_counts}.",
        file)
    ratios = np.array2string(ratios, precision=2, separator=', ')
    augmented_print(f"Class ratios for {dataset_name} set: {ratios}.", file)
    augmented_print(
        f"Number of {dataset_name} samples: {len(dataset)}.", file)


if __name__ == "__main__":
    class Args:
        dataset = 'mnist'


    args = Args()
    set_dataset(args=args)
    num_train = args.num_train_samples
    num_models = 50
    num_samples_per_model = num_train / num_models
    print('num samples per model: ', num_samples_per_model)
