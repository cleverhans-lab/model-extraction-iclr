from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision import datasets
import torch
import os

#def get_imagenet_dataset(args, train=True):
def get_imagenet_dataset(args, split = 'val'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
    # transfrom taken from https://github.com/pytorch/vision/issues/39
    # preprocessing used for all pretained models for torchvision
    if 'mnist' in args.dataset:
        # this means we are attacking a mnist model with imagenet, need to resize, crop and grayscale
        preprocessing.append(transforms.Resize(28))
        preprocessing.append(transforms.CenterCrop(28))
        preprocessing.append(transforms.Grayscale())
    elif 'svhn' in args.dataset:
        # this means we are attacking a mnist model with imagenet, need to resize, crop and grayscale
        preprocessing.append(transforms.Resize(32))
        preprocessing.append(transforms.CenterCrop(32))
    elif 'cifar10' in args.dataset:
        # this means we are attacking a mnist model with imagenet, need to resize, crop and grayscale
        preprocessing.append(transforms.Resize(32))
        preprocessing.append(transforms.CenterCrop(32))

    elif 'imagenet' not in args.dataset:
        # this means we are stealing other model with imagenet data
        # need preprocessing steps
        raise Exception("unimplemented model extraction attack with imagenet data, you might need to add customized preprocessing steps here")
    else:
        # just using imagenet data
        pass
    imagenet_dataset = datasets.ImageNet(
        root="/scratch/ssd002/datasets/imagenet256/",   
        split = split,    #  'val' for validation set, 'train' for training set. By default, 'val' is chosen.
        transform=transforms.Compose(preprocessing)
    )
    return imagenet_dataset