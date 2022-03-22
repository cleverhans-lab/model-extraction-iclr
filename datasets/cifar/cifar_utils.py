from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import os

def get_cifar_private_data(args):
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
            transforms.Normalize(
                (
                    0.49139969,
                    0.48215842,
                    0.44653093),
                (
                    0.24703223,
                    0.24348513,
                    0.26158784)
            )]),
        download=True)
    private_dataset_size = len(all_private_datasets) // args.num_models
    all_private_trainloaders = []
    for i in range(args.num_models):
        begin = i * private_dataset_size
        if i == args.num_models - 1:
            end = len(all_private_datasets)
        else:
            end = (i + 1) * private_dataset_size
        indices = list(range(begin, end))
        private_dataset = Subset(all_private_datasets, indices)
        private_trainloader = DataLoader(private_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, **args.kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders

def get_cifar_dataset(args, train=True):
    if args.attacker_dataset == 'cifar10':
        datasets_cifar = datasets.CIFAR10
        dataset_path = os.path.join(args.path, 'CIFAR10')
    elif args.attacker_dataset == 'cifar100':
        datasets_cifar = datasets.CIFAR100
        dataset_path = os.path.join(args.path, 'CIFAR100')
    else:
        raise Exception(args.datasets_exception)
    normalize = transforms.Normalize(
        mean=[0.49139969,
        0.48215842,
        0.44653093],
        std=[0.24703223,
        0.24348513,
        0.26158784])
    preprocessing = [
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize
    ]
    if 'mnist' in args.dataset:
        # this means we are attacking a mnist model with imagenet, need to resize, crop and grayscale
        preprocessing.append(transforms.Resize(28))
        preprocessing.append(transforms.CenterCrop(28))
        preprocessing.append(transforms.Grayscale())
    dataset = datasets_cifar(root = dataset_path, train=train, transform = transforms.Compose(preprocessing), download = True)
    return dataset


