from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision import datasets, transforms


def get_svhn_train_extra_sets(args):
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
    return trainset, extraset


def get_svhn_private_data(args):
    if args.dataset != 'svhn':
        return None
    trainset, extraset = get_svhn_train_extra_sets(args=args)
    private_trainset_size = len(trainset) // args.num_models
    private_extraset_size = len(extraset) // args.num_models
    all_private_trainloaders = []
    for i in range(args.num_models):
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
        private_dataset = ConcatDataset(
            [Subset(trainset, train_indices),
             Subset(extraset, extra_indices)])
        private_trainloader = DataLoader(
            private_dataset,
            batch_size=args.batch_size,
            shuffle=True, **args.kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders

#without extraset
# def get_svhn_private_data(args):
#     if args.dataset != 'svhn':
#         return None
#     trainset, _ = get_svhn_train_extra_sets(args=args)
#     private_trainset_size = len(trainset) // args.num_models
#     all_private_trainloaders = []
#     for i in range(args.num_models):
#         train_begin = i * private_trainset_size
#         if i == args.num_models - 1:
#             train_end = len(trainset)
#         else:
#             train_end = (i + 1) * private_trainset_size
#         train_indices = list(range(train_begin, train_end))
#         private_dataset = Subset(trainset, train_indices)
#         private_trainloader = DataLoader(
#             private_dataset,
#             batch_size=args.batch_size,
#             shuffle=True, **args.kwargs)
#         all_private_trainloaders.append(private_trainloader)
#     return all_private_trainloaders


class FromSVHNtoMNIST:
    """Convert SVHN to MNIST."""

    def __call__(self, img):
        # img = img.convert("RGB")
        # from PIL import Image
        # img.show()
        # out = img[0] * 0.2126 + img[1] * 0.7152 + img[2] * 0.0722
        # L is 8 bit pixels black and white like in MNIST
        img = img.convert('L')
        # img.show()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
