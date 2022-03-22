from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import json
import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Dict
from typing import Optional
from typing import Union
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from sklearn import metrics
from torch import Tensor
from torch.nn import DataParallel
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.optim.lr_scheduler import MultiStepLR
# from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

from datasets.svhn.svhn_utils import FromSVHNtoMNIST
from datasets.cifar.cifar_utils import get_cifar_private_data, get_cifar_dataset
from datasets.deprecated.coco.helper_functions.helper_functions import \
    average_precision
from datasets.deprecated.coco.helper_functions.helper_functions import mAP
from datasets.mnist.mnist_utils import get_mnist_dataset, get_mnist_private_data
from datasets.mnist.mnist_utils import get_mnist_dataset_by_name
from datasets.mnist.mnist_utils import get_mnist_transforms
from datasets.svhn.svhn_utils import get_svhn_private_data
from datasets.imagenet.dataset_imagenet import get_imagenet_dataset
from models.private_model import get_private_model_by_id
from queryset import QuerySet
from queryset import get_aggregated_labels_filename
from queryset import get_targets_filename
from queryset import get_raw_queries_filename
from queryset import get_queries_filename
from general_utils.save_load import save_obj
from general_utils.functions import sigmoid
import random
import pickle
import socket
from pow.hashcash import mint_iteractive, generate_challenge, check, _to_binary

from ax.service.managed_loop import optimize  # pip install ax-platform


class metric(Enum):
    """
    Evaluation metrics for the models.
    """
    acc = 'acc'
    acc2 = 'acc2' # For fidelity accuracy
    acc_detailed = 'acc_detailed'
    acc_detailed_avg = 'acc_detailed_avg'
    balanced_acc = 'balanced_acc'
    balanced_acc_detailed = 'balanced_acc_detailed'
    auc = 'auc'
    auc_detailed = 'auc_detailed'
    f1_score = 'f1_score'
    f1_score_detailed = 'f1_score_detailed'
    loss = 'loss'
    test_loss = 'test_loss'
    train_loss = 'train_loss'
    map = 'map'
    map_detailed = 'map_detailed'
    gaps_mean = 'gaps_mean'
    gaps_detailed = 'gaps_detailed'
    pc = 'pc'
    rc = 'rc'
    fc = 'fc'
    po = 'po'
    ro = 'ro'
    fo = 'fo'

    def __str__(self):
        return self.name


class result(Enum):
    """
    Properties of the results.
    """
    aggregated_labels = 'aggregated_labels'
    indices_answered = 'indices_answered'
    predictions = 'predictions'
    labels_answered = 'labels_answered'
    count_answered = 'count_answered'

    def __str__(self):
        return self.name


def get_device(args):
    num_devices = torch.cuda.device_count()
    device_ids = args.device_ids
    if not torch.cuda.is_available():
        return torch.device('cpu'), []
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
                .format(num_devices, len(device_ids)))
    if args.cuda:
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')
    return device, device_ids


def get_auc(classification_type, y_true, y_pred, num_classes=None):
    """
    Compute the AUC (Area Under the receiver operator Curve).
    :param classification_type: the type of classification.
    :param y_true: the true labels.
    :param y_pred: the scores or predicted labels.
    :return: AUC score.
    """
    if classification_type == 'binary':
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     y_true, y_pred, pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        auc = metrics.roc_auc_score(
            y_true=y_true,
            y_score=y_pred,
            average='weighted'
        )
    elif classification_type == 'multiclass':
        auc = metrics.roc_auc_score(
            y_true=y_true,
            y_score=y_pred,
            # one-vs-one, insensitive to class imbalances when average==macro
            multi_class='ovo',
            average='macro',
            labels=[x for x in range(num_classes)]
        )
    else:
        raise Exception(
            f"Unexpected classification_type: {classification_type}.")
    return auc


def get_prediction(args, model, unlabeled_dataloader):
    initialized = False
    with torch.no_grad():
        for data, _ in unlabeled_dataloader:
            if args.cuda:
                data = data.cuda()
            output = model(data)
            if not initialized:
                result = output
                initialized = True
            else:
                result = torch.cat((result, output), 0)
    return result

def get_predictionnet(args, model, unlabeled_dataloader):
    """Get predictions using a server client setup with POW on the server side."""
    initialized = False
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432  # The port used by the server
    timequery = 0
    start1 = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (HOST, PORT)
    sock.connect(server_address)
    end1 = time.time()
    timequery+=end1-start1
    try:
        with torch.no_grad():
            for data, _ in unlabeled_dataloader:
                start1 = time.time()
                datastr = pickle.dumps(data)
                sock.sendall(datastr)
                time.sleep(0.1)
                str = "done"
                sock.sendall(str.encode())

                ### POW Challenge
                challenge = sock.recv(4096)
                challenge = pickle.loads(challenge)
                pos = challenge.find(":")
                pos2 = challenge[pos+1:].find(":")
                bits = challenge[pos+1:pos+pos2+1]
                bits = int(bits)
                xtype = 'bin'
                stamp = mint_iteractive(challenge=challenge, bits=bits, xtype=xtype)
                datastamp = pickle.dumps(stamp)
                sock.sendall(datastamp)
                #####

                output = sock.recv(4096)
                output = pickle.loads(output)
                if not initialized:
                    result = output
                    initialized = True
                else:
                    result = torch.cat((result, output), 0)
                end1 = time.time()
                timequery += end1-start1
        start1 = time.time()
        time.sleep(0.1)
        str = "doneiter"
        sock.sendall(str.encode())
        end1 = time.time()
        timequery += end1-start1
    finally:
        sock.close()
    return result, timequery

def count_samples_per_class(dataloader):
    steps = len(dataloader)
    dataiter = iter(dataloader)
    targets = []
    for step in range(steps):
        _, target = next(dataiter)
        if isinstance(target, (int, float)):
            targets.append(target)
        else:
            if isinstance(target, torch.Tensor):
                target = target.detach().cpu().squeeze().squeeze().numpy()
            targets += list(target)
    targets = np.array(targets)
    uniques = np.unique(targets)
    counts = {u: 0 for u in uniques}
    for u in targets:
        counts[u] += 1
    return counts


def get_timestamp():
    dateTimeObj = datetime.now()
    # timestampStr = dateTimeObj.strftime("%Y-%B-%d-(%H:%M:%S.%f)")
    timestampStr = dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S-%f")
    return timestampStr


def get_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = edict(json.load(f))

    user = getpass.getuser()
    for k, v in cfg.items():
        if '{user}' in str(v):
            cfg[k] = v.replace('{user}', user)
    return cfg


def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue

        break

    return lr * np.power(lr_factor, count)


def class_wise_loss_reweighting(beta, samples_per_cls):
    """
     https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab

    :param samples_per_cls: number of samples per class
    :return: weights per class for the loss function
    """
    num_classes = len(samples_per_cls)
    effective_sample_count = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_sample_count)
    # normalize the weights
    weights = weights / np.sum(weights) * num_classes
    return weights


def load_private_data_and_qap(args):
    """Load labeled private data and query-answer pairs for retraining private models."""
    kwargs = get_kwargs(args=args)
    args.kwargs = kwargs
    if 'mnist' in args.dataset:
        all_private_datasets = get_mnist_dataset(args=args, train=True)
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            begin = i * private_dataset_size
            if i == args.num_models - 1:
                end = len(all_private_datasets)
            else:
                end = (i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(
                args,
                transform=get_mnist_transforms(args=args),
                id=i)
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders

    elif args.dataset == 'svhn':
        trainset = datasets.SVHN(
            root=args.dataset_path,
            split='train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.43768212, 0.44376972, 0.47280444),
                    (
                        0.19803013, 0.20101563,
                        0.19703615))]),
            download=True)
        extraset = datasets.SVHN(
            root=args.dataset_path,
            split='extra',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.42997558, 0.4283771, 0.44269393),
                    (0.19630221, 0.1978732, 0.19947216))]),
            download=True)
        private_trainset_size = len(trainset) // args.num_models
        private_extraset_size = len(extraset) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            train_begin = i * private_trainset_size
            extra_begin = i * private_extraset_size
            if i == args.num_models - 1:
                train_end = len(trainset)
            else:
                train_end = (i + 1) * private_trainset_size
            if i == args.num_models - 1:
                extra_end = len(extraset)
            else:
                extra_end = (i + 1) * private_extraset_size
            train_indices = list(range(train_begin, train_end))
            extra_indices = list(range(extra_begin, extra_end))
            private_trainset = Subset(trainset, train_indices)
            private_extraset = Subset(extraset, extra_indices)
            query_dataset = QuerySet(
                args,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.45242317,
                         0.45249586,
                         0.46897715),
                        (0.21943446,
                         0.22656967,
                         0.22850613))]),
                id=i)
            augmented_dataset = ConcatDataset(
                [private_trainset, private_extraset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == 'cifar100':
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        all_private_datasets = datasets_cifar(
            args.dataset_path,
            train=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((
                    0.49139969,
                    0.48215842,
                    0.44653093),
                    (
                        0.24703223,
                        0.24348513,
                        0.26158784))]),
            download=True)
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            begin = i * private_dataset_size
            if i == args.num_models - 1:
                end = len(all_private_datasets)
            else:
                end = (i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(
                args,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.49421429,
                                          0.4851314,
                                          0.45040911),
                                         (0.24665252,
                                          0.24289226,
                                          0.26159238))]),
                id=i)
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    else:
        raise Exception(args.datasets_exception)


def get_data_subset(args, dataset, indices):
    """
    The Subset function differs between datasets, unfortunately.

    :param args: program params
    :param dataset: extract subset of the data from this dataset
    :param indices: the indices in the dataset to be accessed
    :return: the subset
    """
    return Subset(dataset=dataset, indices=indices)


def save_raw_queries_targets(args, dataset, indices, name):
    kwargs = get_kwargs(args=args)
    query_dataset = get_data_subset(args=args, dataset=dataset, indices=indices)

    queryloader = DataLoader(query_dataset, batch_size=args.batch_size,
                             shuffle=False, **kwargs)
    all_samples = []
    all_targets = []
    for data, targets in queryloader:
        all_samples.append(data.numpy())
        all_targets.append(targets.numpy())
    all_samples = np.concatenate(all_samples, axis=0).transpose(0, 2, 3, 1)
    assert len(all_samples.shape) == 4 and all_samples.shape[0] == len(indices)
    all_samples = (all_samples * 255).astype(np.uint8)

    if ('mnist' in args.dataset):
        all_samples = np.squeeze(all_samples)
        shape_len = 3
    else:
        shape_len = 4
    assert len(all_samples.shape) == shape_len

    filename = get_raw_queries_filename(name=name, args=args)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, all_samples)

    save_targets(name=name, args=args, targets=all_targets)


def save_targets(args, name, targets):
    targets = np.concatenate(targets, axis=0)
    filename = get_targets_filename(name=name, args=args)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, targets)


def save_queries(args, dataset, indices, name):
    # Select the query items (data points that) given by indices.
    query_dataset = get_data_subset(args=args, dataset=dataset, indices=indices)

    kwargs = get_kwargs(args=args)
    queryloader = DataLoader(query_dataset, batch_size=args.batch_size,
                             shuffle=False, **kwargs)
    all_samples = []
    all_targets = []
    for data, targets in queryloader:
        all_samples.append(data.numpy())
        all_targets.append(targets.numpy())
    all_samples = np.concatenate(all_samples, axis=0)

    assert len(all_samples.shape) == 4 and all_samples.shape[0] == len(indices)
    if 'mnist' in args.dataset:
        all_samples = np.squeeze(all_samples)
        shape_len = 3
    else:
        shape_len = 4

    assert len(all_samples.shape) == shape_len

    filename = get_queries_filename(name=name, args=args)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, all_samples)

    save_targets(name=name, args=args, targets=all_targets)


def get_all_targets(dataloader) -> Optional[Tensor]:
    dataset = dataloader.dataset
    dataset_len = len(dataset)
    all_targets = None
    with torch.no_grad():
        end = 0
        for _, targets in dataloader:
            batch_size = targets.shape[0]
            begin = end
            end = begin + batch_size
            if all_targets is None:
                if len(targets.shape) == 1:
                    all_targets = torch.zeros(dataset_len)
                if len(targets.shape) == 2:
                    num_labels = targets.shape[1]
                    all_targets = torch.zeros((dataset_len, num_labels))
                else:
                    raise Exception(f"Unknown setting with the shape of "
                                    f"targets: {targets.shape}.")
            all_targets[begin:end] += targets

    return all_targets




def load_private_data(args):
    """Load labeled private data for training private models."""
    kwargs = get_kwargs(args=args)
    args.kwargs = kwargs
    if args.dataset in ['mnist', 'fashion-mnist']:
        return get_mnist_private_data(args=args)
    elif args.dataset == 'svhn':
        return get_svhn_private_data(args=args)
    elif args.dataset.startswith('cifar'):
        return get_cifar_private_data(args=args)

    # return get_cxpert_debug_dataloaders(args=args)
    else:
        raise Exception(args.datasets_exception)


def load_ordered_unlabeled_data(args, indices, unlabeled_dataset):
    """Load unlabeled private data according to a specific order."""
    args.kwargs = get_kwargs(args=args)

    # A part of the original testset is loaded according to a specific order.
    unlabeled_dataset = Subset(unlabeled_dataset, indices)
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **args.kwargs)
    return unlabeled_dataloader


def get_non_trained_set(args):
    """
    This is a previous approach where the unlabeled and test data together
    were kept together. However, it was too entangled.

    :param args:
    :return:
    """
    if 'mnist' in args.dataset:
        dataset = get_mnist_dataset(args=args, train=False)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN(
            root=args.dataset_path,
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.45242317, 0.45249586, 0.46897715),
                    (0.21943446, 0.22656967, 0.22850613))]),
            download=True)
    elif args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == 'cifar100':
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        dataset = datasets_cifar(
            root=args.dataset_path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49421429, 0.4851314, 0.45040911),
                    (0.24665252, 0.24289226,
                     0.26159238))]),
            download=True)
    elif args.dataset == "imagenet":
        dataset = get_imagenet_dataset(args=args, split = 'val')
    else:
        raise Exception(args.datasets_exception)
    return dataset


def get_test_set(args):
    """
    Get the REAL test set. This keeps the unlabeled and test data separately.

    :param args:
    :return: only the test data.
    """
    non_trained_set = get_non_trained_set(args=args)
    if args.attacker_dataset == args.dataset:
        start = args.num_unlabeled_samples
    else:
        # TODO: this number is temporary (for mnist victim model only)
        if args.dataset == "mnist":
            start = 0  # 9000
        else:
            start = args.num_unlabeled_samples
    if args.dataset == "imagenet":
        end = len(non_trained_set)
        indices = random.sample(range(0, end), end-start) #Could be some overlap between indices for querying.
        #indices = random.sample(range(0, end), 10000) # 10000 test elements. Use this when using training set for queries.
        return Subset(dataset=non_trained_set,  # Dont need test set at the moment. Querying happens from get_unlabeled_set
                        indices=indices)
        #print("END", end) 50000
    else:
        end = len(non_trained_set)
    assert end > start
    return Subset(dataset=non_trained_set, indices=list(range(start, end)))


def get_unlabeled_set(args):  # MODIFIED
    """
    Get the REAL unlabeled set.

    :param args:
    :return: only the unlabeled data.
    """

    if args.dataset == 'mnist':
        end = args.num_unlabeled_samples
        # print(args.num_unlabeled_samples)
        dataset = get_mnist_dataset(args=args, train=True)
        subset = Subset(dataset,
                        list(range(50000,
                                   50000 + end)))  # Querying from mnist training set.
    elif args.dataset == "imagenet": # Need to randomly select because the items are in order
        non_trained_set = get_non_trained_set(args=args)
        start = 0
        end = len(non_trained_set)
        #indices = random.sample(range(start, end), args.num_unlabeled_samples)
        subset = Subset(dataset=non_trained_set,
                        indices=list(range(start, end))) # Full test set for querying
    else:
        non_trained_set = get_non_trained_set(args=args)
        start = 0
        end = args.num_unlabeled_samples
        assert end > start
        subset = Subset(dataset=non_trained_set, indices=list(range(start, end)))

    # print(len(subset))
    assert len(subset) == args.num_unlabeled_samples
    return subset


def get_attacker_dataset(args, dataset_name):
    data_dir = args.data_dir
    if 'mnist' in dataset_name:
        dataset = get_mnist_dataset_by_name(args, dataset_name, train=False)
    elif dataset_name == 'svhn':
        svhn_transforms = []
        if 'mnist' in args.dataset:
            # Transform SVHN images from the RGB to L - gray-scale 8 bit images.
            svhn_transforms.append(FromSVHNtoMNIST())
            svhn_transforms.append(transforms.ToTensor())
            # Normalize with the mean and std found for the new images.
            # This closely corresponds to the mean and std of the standard
            # values of mean and std for SVHN.
            svhn_transforms.append(
                transforms.Normalize((0.45771828,), (0.21816934,)))
            svhn_transforms.append(transforms.RandomCrop((28, 28)))
        else:
            svhn_transforms.append(transforms.ToTensor())
            svhn_transforms.append(transforms.Normalize(
                (0.45242317, 0.45249586, 0.46897715),
                (0.21943446, 0.22656967, 0.22850613)))
        dataset_path = os.path.join(data_dir, 'SVHN')
        dataset = datasets.SVHN(
            root=dataset_path,
            split='test',
            transform=transforms.Compose(svhn_transforms),
            download=True)
    elif dataset_name.startswith('cifar'):
        # if dataset_name == 'cifar10':
        #     datasets_cifar = datasets.CIFAR10
        #     dataset_path = os.path.join(args.path, 'CIFAR10')
        # elif dataset_name == 'cifar100':
        #     datasets_cifar = datasets.CIFAR100
        #     dataset_path = os.path.join(args.path, 'CIFAR100')
        # else:
        #     raise Exception(args.datasets_exception)
        # dataset = datasets_cifar(
        #     root=dataset_path,
        #     train=False,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.49421429, 0.4851314, 0.45040911),
        #             (0.24665252, 0.24289226,
        #              0.26159238))]),
        #     download=True)
        dataset = get_cifar_dataset(args)
    elif dataset_name == 'tinyimages':
        from datasets.cifar.tinyimage500k import get_extra_cifar10_data_from_ti
        dataset = get_extra_cifar10_data_from_ti(args=args)
    elif dataset_name == 'imagenet':
        dataset = get_imagenet_dataset(args=args)
    else:
        raise Exception(args.datasets_exception)
    return dataset


def get_unlabeled_standard_indices(args):
    """
    Get the indices for each querying party from the test data.

    :param args: arguments
    :return: indices for each querying party
    """
    data_indices = [[] for _ in args.querying_parties]
    num_querying_parties = len(args.querying_parties)
    # Only a part of the original test set is used for the query selection.
    size = args.num_unlabeled_samples // num_querying_parties
    for i in range(num_querying_parties):
        begin = i * size
        # Is it the last querying party?
        if i == num_querying_parties - 1:
            end = args.num_unlabeled_samples
        else:
            end = (i + 1) * size
        indices = list(range(begin, end))
        data_indices[i] = indices
    return data_indices


def get_unlabeled_indices(args, dataset):
    data_indices = get_unlabeled_standard_indices(args=args)

    num_querying_parties = len(args.querying_parties)
    # Test correctness of the computed indices by summations.
    assert sum(
        [len(data_indices[i]) for i in
         range(num_querying_parties)]) == args.num_unlabeled_samples
    assert len(
        set(np.concatenate(data_indices, axis=0))) == args.num_unlabeled_samples

    return data_indices


def load_unlabeled_dataloaders(args, unlabeled_dataset=None):
    """
    Load unlabeled private data for query selection.
    :return: all_unlabeled_dataloaders data loaders for each querying party
    """
    kwargs = get_kwargs(args=args)

    all_unlabeled_dataloaders = []

    if unlabeled_dataset is None:
        unlabeled_dataset = get_unlabeled_set(args=args)

    unlabeled_indices = get_unlabeled_indices(args=args,
                                              dataset=unlabeled_dataset)
    # Create data loaders.
    for indices in unlabeled_indices:
        unlabeled_dataset = Subset(unlabeled_dataset, indices)
        unlabeled_dataloader = DataLoader(
            unlabeled_dataset,
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
        all_unlabeled_dataloaders.append(unlabeled_dataloader)
    return all_unlabeled_dataloaders


def get_kwargs(args):
    kwargs = {'num_workers': args.num_workers,
              'pin_memory': True} if args.cuda else {}
    return kwargs


def load_training_data(args):
    """Load labeled data for training non-private baseline models."""
    kwargs = get_kwargs(args=args)
    if 'mnist' in args.dataset:
        trainset = get_mnist_dataset(args=args, train=True)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 shuffle=True, **kwargs)
    elif args.dataset == 'svhn':
        trainset = datasets.SVHN(root=args.dataset_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.43768212, 0.44376972, 0.47280444),
                                         (
                                             0.19803013, 0.20101563,
                                             0.19703615))]),
                                 download=True)
        # extraset = datasets.SVHN(root=args.dataset_path,
        #                          split='extra',
        #                          transform=transforms.Compose([
        #                              transforms.ToTensor(),
        #                              transforms.Normalize(
        #                                  (0.42997558, 0.4283771, 0.44269393),
        #                                  (0.19630221, 0.1978732, 0.19947216))]),
        #                          download=True)
        # trainloader = DataLoader(ConcatDataset([trainset, extraset]),
        #                          batch_size=args.batch_size, shuffle=True,
        #                          **kwargs)
        trainloader = DataLoader(trainset,
                                 batch_size=args.batch_size, shuffle=True,
                                 **kwargs)
    elif args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == 'cifar100':
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        trainset = datasets_cifar(
            args.dataset_path,
            train=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139969,
                     0.48215842,
                     0.44653093),
                    (0.24703223,
                     0.24348513,
                     0.26158784))]),
            download=True)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 shuffle=True, **kwargs)
    elif args.dataset == "imagenet":
        trainset = get_imagenet_dataset(args=args, split = 'train')
        end = len(trainset)
        indices = random.sample(range(0, end), 10000)
        #train_set = Subset(dataset=trainset, indices=list(range(end-10000, end)))
        train_set = Subset(dataset=trainset,
                           indices=indices)
        trainloader = DataLoader(train_set, batch_size=args.batch_size,
                                 shuffle=True, **kwargs)
    else:
        raise Exception(args.datasets_exception)
    return trainloader


def load_evaluation_dataloader(args):  # Modified
    """Load labeled data for evaluation."""
    kwargs = get_kwargs(args=args)
    dataset = get_test_set(args=args)
    evalloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            **kwargs)
    return evalloader


def load_unlabeled_dataloader(args):
    """Load all unlabeled data."""
    kwargs = get_kwargs(args=args)
    dataset = get_unlabeled_set(args=args)
    unlabeled_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=False,
                                  **kwargs)
    return unlabeled_loader


def get_unlabled_last_index(
        num_unlabeled_samples,
        len_dataset,
        len_class):
    """
    For example, for CIFAR100 we have len_dataset 10000 for test data.
    The number of samples per class is 1000.
    If we want 9000 unlabeled samples then the ratio_unlabeled is 9/10.
    The number of samples per class for the unlabeled dataset is 9/10*100=90.
    If the number of samples for the final test is 1000 samples and we have 100
    classes, then the number of samples per class will be 10 (only).

    :param num_unlabeled_samples: number of unlabeled samples from the test set
    :param len_dataset: the total number of samples in the intial test set
    :param len_class: the number of samples for a given class
    :return: for the array of sample indices for the class, the last index for
    the unlabeled part

    >>> num_unlabeled_samples = 9000
    >>> len_dataset = 10000
    >>> len_class = 100
    >>> result = get_unlabled_last_index(num_unlabeled_samples=num_unlabeled_samples, len_dataset=len_dataset, len_class=len_class)
    >>> assert result == 90
    >>> # print('result: ', result)
    """
    ratio_unlabeled = num_unlabeled_samples / len_dataset
    last_unlabeled_index = int(ratio_unlabeled * len_class)
    return last_unlabeled_index


def regularize_loss(model):
    loss = 0
    for param in list(model.children())[0].parameters():
        loss += 2e-5 * torch.sum(torch.abs(param))
    return loss


def get_loss_criterion(model, args):
    """
    Get the loss criterion.

    :param model: model
    :param args: arguments
    :return: the loss criterion (function like to be called)
    """
    if args.loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss_type == 'BCE':
        criterion = nn.BCELoss()
    elif args.loss_type == 'BCEWithLogits':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Unknown loss type: {args.loss_type}.")

    return criterion


def pure_model(model):
    """
    Extract the proper model if enclosed in DataParallel (distributed model
    feature).

    :param model: a model
    :return: pure PyTorch model
    """
    if hasattr(model, 'module'):
        return model.module
    else:
        return model




def task_loss(target, output, criterion, weights):
    """
    Compute the loss per task / label.

    :param target: target labels
    :param output: predicted labels
    :param criterion: loss criterion
    :param weights: the weight per task / label
    :return: the computed loss
    """
    loss = torch.zeros(1).to(output.device).to(torch.float32)
    for task in range(target.shape[1]):
        task_output = output[:, task]
        task_target = target[:, task]
        mask = ~torch.isnan(task_target)
        task_output = task_output[mask]
        task_target = task_target[mask]
        if len(task_target) > 0:
            task_loss = criterion(task_output.float(), task_target.float())
            if weights is None:
                loss += task_loss
            else:
                loss += weights[task] * task_loss

    return loss


def compute_loss(target, output, criterion, weights, args, model, data):
    """
    Compute the loss.

    :param target: target labels
    :param output: predicted labels
    :param criterion: loss criterion
    :param weights: the weight per task / label
    :return: the computed loss
    """
    loss = criterion(output, target)
    return loss


def train(model, trainloader, optimizer, criterion, args):
    """Train a given model on a given dataset using a given optimizer,
    loss criterion, and other arguments, for one epoch."""
    model.train()
    losses = []

    device, _ = get_device(args=args)

    weights = None

    for batch_id, (data, target) in enumerate(trainloader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.loss_type in {'MSE', 'BCE', 'BCEWithLogits'}:
            data = data.to(torch.float32)
            target = target.to(torch.float32)
        else:
            target = target.to(torch.long)

        optimizer.zero_grad()
        output = model(data)

        loss = compute_loss(target=target, output=output, criterion=criterion,
                            weights=weights, args=args, model=model, data=data)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    train_loss = np.mean(losses)
    return train_loss


def evaluate_multiclass(model, dataloader, args, victimmodel = None):
    """
    Evaluation for standard multiclass classification.
    Evaluate metrics such as accuracy, detailed acc, balanced acc, auc of a given model on a given dataset.

    Accuracy detailed - evaluate the class-specific accuracy of a given model on a given dataset.

    :return:
    detailed_acc: A 1-D numpy array of length L = num-classes, containing the accuracy for each class.

    """
    model.eval()
    losses = []
    correct = 0
    correct2 = 0
    total = len(dataloader.dataset)
    correct_detailed = np.zeros(args.num_classes, dtype=np.int64)
    wrong_detailed = np.zeros(args.num_classes, dtype=np.int64)
    raw_softmax = None
    raw_preds = []
    raw_targets = []
    criterion = get_loss_criterion(model=model, args=args)
    with torch.no_grad():
        for data, target in dataloader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(input=output, target=target)
            losses.append(loss.item())

            preds = output.data.argmax(axis=1)
            labels = target.data.view_as(preds)
            if victimmodel == None:
                correct += preds.eq(labels).cpu().sum().item()
                acc2 = 0 # Fidelity accuracy not used
            else:
                correct += preds.eq(labels).cpu().sum().item()
                output2 = victimmodel(data)
                preds2 = output2.data.argmax(axis=1)
                correct2 += preds2.eq(preds).cpu().sum().item() # For jacobian calculate acc against victim model as well
                acc2 = 100. * correct2 / total
            softmax_outputs = F.softmax(output, dim=1)
            softmax_outputs = softmax_outputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            preds_np = preds.detach().cpu().numpy().astype(int)

            if raw_softmax is None:
                raw_softmax = softmax_outputs
            else:
                raw_softmax = np.append(raw_softmax, softmax_outputs, axis=0)
            raw_targets = np.append(raw_targets, labels_np)
            raw_preds = np.append(raw_preds, preds_np)

            for label, pred in zip(target, preds):
                if label == pred:
                    correct_detailed[label] += 1
                else:
                    wrong_detailed[label] += 1

    loss = np.mean(losses)

    acc = 100. * correct / total

    balanced_acc = metrics.balanced_accuracy_score(
        y_true=raw_targets,
        y_pred=raw_preds,
    )

    if (np.round(raw_softmax.sum(axis=1)) == 1).all() and raw_targets.size > 0:
        try:
            auc = get_auc(
                classification_type=args.class_type,
                y_true=raw_targets,
                y_pred=raw_softmax,
                num_classes=args.num_classes,
            )
        except ValueError as err:
            print('Error occurred: ', err)
            # Transform to list to print the full array.
            print('y_true: ', raw_targets.tolist())
            print('y_pred: ', raw_softmax.tolist())
            auc = 0
    else:
        auc = 0

    assert correct_detailed.sum() + wrong_detailed.sum() == total
    acc_detailed = 100. * correct_detailed / (correct_detailed + wrong_detailed)

    mAP_score = mAP(
        targs=one_hot_numpy(raw_targets.astype(np.int), args.num_classes),
        preds=one_hot_numpy(raw_preds.astype(np.int), args.num_classes))

    result = {
        metric.loss: loss,
        metric.acc: acc,
        metric.acc2: acc2,
        metric.balanced_acc: balanced_acc,
        metric.auc: auc,
        metric.acc_detailed: acc_detailed,
        metric.map: mAP_score,
    }

    return result

def get_metrics_processed(values):
    values = np.array(values)
    values = values[~np.isnan(values)]
    value = np.mean(values)
    return value, values


def one_hot(indices, num_classes: int) -> Tensor:
    """
    Convert labels into one-hot vectors.

    Args:
        indices: a 1-D vector containing labels.
        num_classes: number of classes.

    Returns:
        A 2-D matrix containing one-hot vectors, with one vector per row.
    """
    onehot = torch.zeros((len(indices), num_classes))
    for i in range(len(indices)):
        onehot[i][indices[i]] = 1
    return onehot


def one_hot_numpy(indices, num_classes: int) -> Tensor:
    """
    Convert labels into one-hot vectors.

    Args:
        indices: a 1-D vector containing labels.
        num_classes: number of classes.

    Returns:
        A 2-D matrix containing one-hot vectors, with one vector per row.
    """
    return one_hot(indices=indices, num_classes=num_classes).numpy()


def augmented_print(text, file, flush: bool = False) -> None:
    """Print to both the standard output and the given file."""
    assert isinstance(text, str)
    print(text)
    if isinstance(file, str):
        openfile = open(file, "a")
        openfile.write(text + "\n")
        if flush:
            sys.stdout.flush()
            openfile.flush()
        openfile.close()
    else:
        file.write(text + "\n")
        if flush:
            sys.stdout.flush()
            file.flush()


def extract_metrics(inputs):
    """
    Get only the keys and value for metrics.

    :param inputs: a dict
    :return: dict with metrics only
    """
    metric_keys = set(metric)
    outputs = {}
    for key, value in inputs.items():
        if key in metric_keys:
            outputs[key] = value
    return outputs


def from_result_to_str(
        result: Dict[metric, Union[int, float, str, np.ndarray, list]],
        sep: str = ';', inner_sep=';') -> str:
    """
    Transform the result in a form of a dict to a pretty string.

    :param result: result in a form of a dict
    :param sep: separator between key-value pairs
    :param inner_sep: separator between keys and values
    :return: pretty string
    """
    out = ""
    for key, value in result.items():
        if value is not None:
            out += str(key) + inner_sep
            out += get_value_str(value=value)
            out += sep
    return out


def get_value_str(value, separator=','):
    if isinstance(value, (int, float, str)):
        out = str(value)
    else:
        # print(__file__ + ' key: ', key)
        out = np.array2string(value, precision=4, separator=separator)
    return out


def update_summary(summary: dict, result: dict) -> dict:
    """
    Append values from result to summary.

    :param summary: summary dict (with aggregated results)
    :param result: result dict (with partial results)
    :return: the updated summary dict
    >>> a = {'a': 1, 'b': 2}
    >>> b = {'a': [3]}
    >>> c = update_summary(summary=b, result=a)
    >>> assert c['a'] == [3, 1]
    """
    for key, value in summary.items():
        if key in result.keys():
            new_value = result[key]
            summary[key].append(new_value)
    return summary


def class_ratio(dataset, args):
    """The ratio of each class in the given dataset."""
    counts = np.zeros(args.num_classes, dtype=np.int64)
    total_count = 0
    for data_index in range(len(dataset)):
        # Get dataset item.
        data_item = dataset[data_index]
        # Get labels.
        label = data_item[1]
        if args.class_type in ['multiclass', 'binary']:
            index = int(label)
            counts[index] += 1
            total_count += 1
        else:
            raise Exception(f"Unknown class type: {args.class_type}.")

    assert counts.sum() == total_count
    return counts, 100. * counts / len(dataset)


def get_class_indices(dataset, args):
    """The indices of samples belonging to each class."""
    indices = [[] for _ in range(args.num_classes)]
    for i in range(len(dataset)):
        # dataset_i = dataset[i]
        # print('dataset[i]: ', dataset_i)
        # dataset_i_1 = dataset_i[1]
        # print('dataset[i][1]: ', dataset_i_1)
        # add index for a given class
        indices[dataset[i][1]].append(i)
    indices = [np.asarray(indices[i]) for i in range(args.num_classes)]

    # Double assert below to check if the number of collected indices for each class add up to total length of the dataset.
    assert sum([len(indices[i]) for i in range(args.num_classes)]) == len(
        dataset)
    assert len(set(np.concatenate(indices, axis=0))) == len(dataset)

    return indices


def get_scheduler(args, optimizer, trainloader=None):
    scheduler_type = args.scheduler_type
    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode='min',
            factor=args.schedule_factor,
            patience=args.schedule_patience)
    elif scheduler_type == 'MultiStepLR':
        milestones = args.scheduler_milestones
        if milestones is None:
            milestones = [int(args.num_epochs * 0.5),
                          int(args.num_epochs * 0.75),
                          int(args.num_epochs * 0.9)]
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=milestones,
            gamma=args.schedule_factor)
    elif scheduler_type == 'custom':
        scheduler = None
    else:
        raise Exception("Unknown scheduler type: {}".format(scheduler_type))
    return scheduler


def get_optimizer(params, args, lr=None):
    if lr is None:
        lr = args.lr
    if args.optimizer == 'SGD':
        return SGD(params, lr=lr,
                   momentum=args.momentum,
                   weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        return Adadelta(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adagrad':
        return Adagrad(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        return Adam(params, lr=lr, weight_decay=args.weight_decay,
                    amsgrad=args.adam_amsgrad)
    elif args.optimizer == 'RMSprop':
        return RMSprop(params, lr=lr, momentum=args.momentum,
                       weight_decay=args.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(args.optimizer))


def distribute_model(args, model):
    device, device_ids = get_device(args=args)
    model = DataParallel(model, device_ids=device_ids).to(device)
    return model


def eval_distributed_model(args, model, dataloader):
    model = distribute_model(args=args, model=model)
    return eval_model(args=args, model=model, dataloader=dataloader)


def eval_model(args, model, dataloader, victimmodel = None):
    result = evaluate_multiclass(
        model=model, dataloader=dataloader, args=args, victimmodel=victimmodel)
    return result


def get_model_params(model, args):
    return model.parameters()


def train_model(args, model, trainloader, evalloader, patience=None):
    device, device_ids = get_device(args=args)
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    model_params = get_model_params(model=model, args=args)
    optimizer = get_optimizer(params=model_params, args=args)
    scheduler = get_scheduler(args=args, optimizer=optimizer,
                              trainloader=trainloader)
    criterion = get_loss_criterion(model=model, args=args)
    if patience is not None:
        # create variables for the patience mechanism
        best_loss = None
        patience_counter = 0

    start_epoch = 0
    save_model_path = getattr(args, 'save_model_path', None)
    if save_model_path is not None:
        filename = "checkpoint-1.pth.tar"  # .format(model.module.name)
        filepath = os.path.join(save_model_path, filename)

        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            print(
                'Restarted from checkpoint file {} at epoch {}'.format(filepath,
                                                                       start_epoch))
    print('STARTED TRAINING')
    for epoch in range(start_epoch, args.num_epochs):
        start = time.time()
        train_loss = train(model=model, trainloader=trainloader, args=args,
                           optimizer=optimizer, criterion=criterion)
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        if args.scheduler_type == 'OneCycleLR':
            scheduler.step()
        else:
            scheduler.step(train_loss)
        stop = time.time()
        epoch_time = stop - start

        if patience is not None:
            result_test = train_model_log(
                args=args, epoch=epoch, model=model, epoch_time=epoch_time,
                trainloader=trainloader, evalloader=evalloader)
            if result_test is None:
                raise Exception(
                    "Fatal Error, result should not be None after training model")
            if best_loss is None or best_loss < result_test[metric.loss]:
                best_loss = result_test[metric.loss]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            train_model_log(
                args=args, epoch=epoch, model=model, epoch_time=epoch_time,
                trainloader=trainloader, evalloader=evalloader)


def bayesian_optimization_training_loop(
        args,
        model,
        adaptive_dataset,
        evalloader,
        patience=10,
        num_optimization_loop=20):
    train_model_function = train_with_bayesian_optimization(
        args,
        adaptive_dataset,
        evalloader,
        patience=patience)

    best_parameters, values, _, _ = optimize(
        parameters=[
            {"name": "lr", "type": "range", "log_scale": True,
             "bounds": [args.lr / 1000, args.lr * 10]},
            {"name": "batch_size", "type": "range", "value_type": "int",
             "log_scale": True, "bounds": [max(1, int(args.batch_size / 16)),
                                           max(int(args.batch_size * 16),
                                               128)]},
        ],
        objective_name="val_loss",
        evaluation_function=train_model_function,
        minimize=True,
        total_trials=num_optimization_loop)

    return best_parameters


def train_with_bayesian_optimization(args, adaptive_dataset, evalloader,
                                     patience=10):
    device, device_ids = get_device(args=args)

    def train_model_wrapper_for_bayesian_optimization(parameters):
        # create a new model
        model = get_private_model_by_id(args=args, id=0)
        model = DataParallel(model, device_ids=device_ids).to(device).train()
        model_params = get_model_params(model=model, args=args)
        lr = parameters.get("lr")
        batch_size = parameters.get("batch_size")

        trainloader = DataLoader(
            adaptive_dataset,
            batch_size=batch_size,
            shuffle=False,
            **args.kwargs)

        optimizer = get_optimizer(params=model_params, args=args, lr=lr)
        scheduler = get_scheduler(args=args, optimizer=optimizer,
                                  trainloader=trainloader)
        criterion = get_loss_criterion(model=model, args=args)

        # create variables for the patience mechanism
        best_loss = None
        patience_counter = 0
        result_test = {}

        start_epoch = 0
        save_model_path = getattr(args, 'save_model_path', None)
        if save_model_path is not None:
            filename = "checkpoint-1.pth.tar"  # .format(model.module.name)
            filepath = os.path.join(save_model_path, filename)

            if os.path.exists(filepath) and args.retrain_extracted_model:
                try:
                    checkpoint = torch.load(filepath)
                    if hasattr(model, 'module'):
                        model.module.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint['state_dict'])
                    print(
                        'Restarted from checkpoint file {} at iteration {}'.format(
                            filepath,
                            checkpoint['epoch']))
                except Exception:
                    print(
                        "find trained model but cannot read, may be a model from previous generation of code")
                    print("train from scratch instead")
        print('STARTED TRAINING')
        for epoch in range(0, args.num_epochs):
            start = time.time()
            train_loss = train(model=model, trainloader=trainloader, args=args,
                               optimizer=optimizer, criterion=criterion)
            # Scheduler step is based only on the train data, we do not use the
            # test data to schedule the decrease in the learning rate.
            if args.scheduler_type == 'OneCycleLR':
                scheduler.step()
            else:
                scheduler.step(train_loss)
            stop = time.time()
            epoch_time = stop - start

            result_test = train_model_log(
                args=args, epoch=epoch, model=model, epoch_time=epoch_time,
                trainloader=trainloader, evalloader=evalloader)
            if not result_test:
                raise Exception(
                    "Fatal Error, result should not be None after training model")

            if epoch == 0:
                # record best loss
                best_loss = result_test[metric.loss]
            if patience is not None:
                if best_loss > result_test[metric.loss]:
                    patience_counter = 0
                    best_loss = result_test[metric.loss]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        balanced_acc = result_test.get(metric.balanced_acc, 0)
        print(
            f"lr:{lr}, "
            f"bs:{batch_size}, "
            f"val_loss:{best_loss}, "
            f"balanced acc:{balanced_acc}"
        )

        return best_loss

    return train_model_wrapper_for_bayesian_optimization


def train_with_bayesian_optimization_with_best_hyperparameter(
        args,
        model,
        adaptive_dataset,
        evalloader,
        parameters,
        patience=10):
    device, device_ids = get_device(args=args)

    model = DataParallel(model, device_ids=device_ids).to(device).train()
    model_params = get_model_params(model=model, args=args)
    lr = parameters.get("lr")
    batch_size = parameters.get("batch_size")

    trainloader = DataLoader(
        adaptive_dataset,
        batch_size=batch_size,
        shuffle=False,
        **args.kwargs)

    optimizer = get_optimizer(params=model_params, args=args, lr=lr)
    scheduler = get_scheduler(args=args, optimizer=optimizer,
                              trainloader=trainloader)
    criterion = get_loss_criterion(model=model, args=args)

    # create variables for the patience mechanism
    best_loss = None
    patience_counter = 0
    result_test = {}

    start_epoch = 0
    save_model_path = getattr(args, 'save_model_path', None)
    if save_model_path is not None:
        filename = "checkpoint-1.pth.tar"  # .format(model.module.name)
        filepath = os.path.join(save_model_path, filename)

        if os.path.exists(filepath) and args.retrain_extracted_model:
            try:
                checkpoint = torch.load(filepath)
                if hasattr(model, 'module'):
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'])
                print(
                    'Restarted from checkpoint file {} at iteration {}'.format(
                        filepath,
                        checkpoint['epoch']))
            except Exception:
                print(
                    "find trained model but cannot read, may be a model from previous generation of code")
                print("train from scratch instead")
    print('STARTED TRAINING')
    for epoch in range(0, args.num_epochs):
        start = time.time()
        train_loss = train(model=model, trainloader=trainloader, args=args,
                           optimizer=optimizer, criterion=criterion)
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        if args.scheduler_type == 'OneCycleLR':
            scheduler.step()
        else:
            scheduler.step(train_loss)
        stop = time.time()
        epoch_time = stop - start

        result_test = train_model_log(
            args=args, epoch=epoch, model=model, epoch_time=epoch_time,
            trainloader=trainloader, evalloader=evalloader)
        if not result_test:
            raise Exception(
                "Fatal Error, result should not be None after training model")

        if epoch == 0 or best_loss < result_test[metric.loss]:
            # record best loss
            best_loss = result_test[metric.loss]
        if patience is not None:
            if best_loss < result_test[metric.loss]:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    balanced_acc = result_test.get(metric.balanced_acc, 0)
    print(
        f"lr:{lr}, "
        f"bs:{batch_size}, "
        f"val_loss:{best_loss}, "
        f"balanced acc:{balanced_acc}"
    )
    return model



def train_model_log(args, epoch, epoch_time, model, trainloader, evalloader):
    log_every = args.log_every_epoch
    print('EPOCH: ', epoch)
    if log_every != 0 and epoch % log_every == 0:
        start_time = time.time()
        result_train = eval_model(model=model, dataloader=trainloader,
                                  args=args)
        result_test = eval_model(model=model, dataloader=evalloader,
                                 args=args)
        stop_time = time.time()
        eval_time = stop_time - start_time
        if epoch == 0:
            header = ['epoch',
                      'train_' + str(metric.loss),
                      'test_' + str(metric.loss),
                      'train_' + str(metric.balanced_acc),
                      'test_' + str(metric.balanced_acc),
                      'train_' + str(metric.auc),
                      'test_' + str(metric.auc),
                      'train_' + str(metric.map),
                      'test_' + str(metric.map),
                      'eval_time',
                      'epoch_time',
                      ]
            header_str = args.sep.join(header)
            print(header_str)
            best_loss = result_test[metric.loss]
        data = [
            epoch,
            result_train[metric.loss],
            result_test[metric.loss],
            result_train[metric.balanced_acc],
            result_test[metric.balanced_acc],
            result_train[metric.auc],
            result_test[metric.auc],
            result_train[metric.map],
            result_test[metric.map],
            eval_time,
            epoch_time,
        ]
        data_str = args.sep.join([str(f"{x:.4f}") for x in data])
        print(data_str)



    try:
        return result_test
    except NameError:
        return eval_model(model=model, dataloader=evalloader,
                          args=args)


def pick_labels_cols(
        target_labels_index: List[int],
        labels: np.array) -> np.array:
    assert len(labels.shape) > 1
    target_labels = []
    for idx in target_labels_index:
        target_labels.append(labels[:, idx])
    target_labels = np.swapaxes(target_labels, 0, 1)
    return target_labels


def retain_labels_cols(
        target_labels_index: List[int],
        labels: np.array) -> np.array:
    assert len(labels.shape) > 1
    num_cols = labels.shape[1]
    target_labels_index = set(target_labels_index)
    for idx in range(num_cols):
        if idx not in target_labels_index:
            labels[:, idx] = np.nan
    return labels


def pick_labels_rows(
        target_labels_index: List[int],
        labels: Dict[int, np.array]) -> Dict[int, np.array]:
    target_labels = {}
    for target_idx, label_idx in enumerate(target_labels_index):
        target_labels[target_idx] = labels[label_idx]
    return target_labels


def print_metrics_detailed(results):
    """
    Print the 4 main metrics detailed in 4 columns.
    :param results: the results with
    """
    arrays = []
    for metric_key in [metric.acc_detailed,
                       metric.balanced_acc_detailed,
                       metric.auc_detailed,
                       metric.map_detailed]:
        arrays.append(results.get(metric_key, None))
    arrays = [x for x in arrays if x is not None]
    expanded_arrays = []
    for array in arrays:
        expanded_arrays.append(np.expand_dims(array, 1))
    all = np.concatenate(expanded_arrays, axis=1)
    print('Print for each label separately: ')
    print('acc,bac,auc,map')
    all_str = "\n".join(','.join(str(y) for y in x) for x in all)
    print(all_str)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
