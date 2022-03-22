from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision import datasets
import torch
import os


def get_mnist_transforms(args, dataset=None):
    if dataset is None:
        dataset = args.dataset
    transformations = []
    if dataset == 'mnist':
        transformations.append(transforms.ToTensor())
        transformations.append(
            transforms.Normalize((0.13251461,), (0.31048025,)))
    elif dataset == 'fashion-mnist':
        transformations.append(transforms.ToTensor())
    elif dataset == 'kmnist':
        transformations.append(transforms.ToTensor())
        transformations.append(
            transforms.Normalize((0.1932,), (0.3499,)))
    elif dataset == 'emnist_all':
        transformations.append(transforms.ToTensor())
        transformations.append(
            transforms.Normalize((0.1840,), (0.3381,)))
    elif dataset == 'emnist':
        transformations.append(transforms.ToTensor())
        transformations.append(
            transforms.Normalize((0.2,), (0.3499,)))
    else:
        raise Exception(args.datasets_exception)
    return transforms.Compose(transformations)


def get_mnist_dataset(args, train=True, dataset=None):   # MODIFIED
    if dataset is None:
        dataset = args.dataset
    # used for attacker dataset
    transform = get_mnist_transforms(args=args, dataset=dataset)
    if dataset == 'mnist':
        dataset_path = os.path.join(args.path, 'MNIST')
        dataset_extractor = datasets.MNIST
    elif dataset == 'fashion-mnist':
        dataset_path = os.path.join(args.path, 'Fashion-MNIST')
        dataset_extractor = datasets.FashionMNIST
    elif dataset == 'kmnist':
        dataset_path = os.path.join(args.path, 'KMNIST')
        dataset_extractor = datasets.KMNIST
    elif dataset == 'emnist' or dataset == 'emnist_all':
        dataset_path = os.path.join(args.path, 'EMNIST')
        dataset_extractor = datasets.EMNIST
        if 'all' in dataset:
            mnist_dataset = dataset_extractor(
                root=dataset_path,
                train=train,
                transform=transform,
                split="balanced",
                download=True)
        else:
            mnist_dataset = dataset_extractor(
                root=dataset_path,
                train=train,
                transform=transform,
                split="letters",
                download=True)
        args.num_unlabeled_samples = mnist_dataset.data.size()[0]
        return mnist_dataset
    else:
        raise Exception(args.datasets_exception)
    mnist_dataset = dataset_extractor(
        root=dataset_path,
        train=train,
        transform=transform,
        download=True)
    #args.num_unlabeled_samples = mnist_dataset.data.size()[0]
    if dataset == "mnist":
        args.num_unlabeled_samples = 10000 # We are using 10000 examples.
    return mnist_dataset


def get_mnist_dataset_by_name(args, dataset, train=True):
    transform = get_mnist_transforms(args=args)
    if dataset == 'mnist':
        dataset_path = os.path.join(args.path, 'MNIST')
        dataset_extractor = datasets.MNIST
    elif dataset == 'fashion-mnist':
        dataset_path = os.path.join(args.path, 'Fashion-MNIST')
        dataset_extractor = datasets.FashionMNIST
    else:
        raise Exception(args.datasets_exception)
    mnist_dataset = dataset_extractor(
        root=dataset_path,
        train=train,
        transform=transform,
        download=True)
    return mnist_dataset


def get_mnist_private_data(args):   # Modified
    if args.num_models == 1 and args.commands == 'basic_model_stealing_attack':
        split = True
    else:
        split = False
    if not 'mnist' in args.dataset:
        return None
    all_private_datasets = get_mnist_dataset(args=args, train=True)
    private_dataset_size = len(all_private_datasets) // args.num_models
    all_private_trainloaders = []
    if args.shuffle_dataset:
        random_indices = torch.randperm(len(all_private_datasets))
    for i in range(args.num_models):
        begin = i * private_dataset_size
        if i == args.num_models - 1:
            end = len(all_private_datasets)
        else:
            end = (i + 1) * private_dataset_size
        if split == True:
            end = 50000
        indices = list(range(begin, end))
        if args.shuffle_dataset:
            indices = random_indices[indices].tolist()
        private_dataset = Subset(all_private_datasets, indices)
        private_trainloader = DataLoader(
            private_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **args.kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders
