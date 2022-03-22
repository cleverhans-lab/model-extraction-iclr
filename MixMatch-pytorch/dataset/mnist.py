import numpy as np
from PIL import Image

import torchvision
import torch


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

# Might have to convert data to rgb :
#transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
#Otherwise may have to change the model to MnistNEtPAte (probably simpler)



def get_mnist(root, n_labeled,
                transform_train=None, transform_val=None,
                download=True):
    base_dataset = torchvision.datasets.MNIST(root, train=True, download=download)
    test_dataset = torchvision.datasets.CIFAR10(root, train=False,
                                                download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(test_dataset.targets, int(n_labeled / 10)) #was base_dataset

    train_labeled_dataset = MNIST_labeledmod(root, train_labeled_idxs, train=False, transform=transform_train) # True
    train_unlabeled_dataset = MNIST_unlabeled(root, train_unlabeled_idxs, train=False,
                                                transform=TransformTwice(transform_train))
    val_dataset = MNIST_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = MNIST_labeled(root, val_idxs, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)} #Test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, val_idxs


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    # To get an equal number of samples per class.
    # for i in range(10):
    #     idxs = np.where(labels == i)[0]
    #     np.random.shuffle(idxs)
    #     train_labeled_idxs.extend(idxs[:n_labeled_per_class])
    #     train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
    #     val_idxs.extend(idxs[-500:])

    # Random selection for points:
    n_labeled = n_labeled_per_class * 10
    idxs = np.where(labels < 10)[0] # All points
    np.random.shuffle(idxs)
    train_labeled_idxs.extend(idxs[:n_labeled])
    train_unlabeled_idxs.extend(idxs[n_labeled: -1000])
    val_idxs.extend(idxs[-1000:])
    ent = 0
    gap = 0
    temp1 = np.load("mnistent.npy")
    temp2 = np.load("mnistgap.npy")
    for i in train_labeled_idxs:
        ent += temp1[i]
        gap += temp2[i]
    # pknn = 0
    total = n_labeled_per_class * 10
    file = f"mnist@{total}new/stats.txt"
    f = open(file, "w")
    f.write("Entropy: " + str(ent) + "\n")
    f.write("Gap: " + str(gap) + "\n")
    f.close()


    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


mnist_mean =  (0.1325)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
mnist_std = (0.3105)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=mnist_mean, std=mnist_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([0,2,1]) # ????


def pad(x, border=4):  # Not working fine
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class MNIST_labeled(torchvision.datasets.MNIST):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(MNIST_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            #print("targets", self.targets)
        #self.data = transpose(normalise(self.data))
        self.data = normalise(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index] # img, target are numpy array
        img = img.reshape((1,28,28))
        img = np.repeat(img, 3, axis = 0)
        #print(img.shape)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNIST_unlabeled(MNIST_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(MNIST_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])


class MNIST_labeledmod(torchvision.datasets.MNIST):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(MNIST_labeledmod, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            #victim = load_private_model_by_id()
            # temp = []
            # for i in indexs:
            #self.targets = victim(self.data)
            # Use model predictions here?
            targets = np.load("mnisttargets.npy")
            self.targets = np.array(targets)[indexs]
            #print("targets", self.targets)
        self.data = normalise(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.reshape((1,28,28))
        img = np.repeat(img, 3, axis = 0)
        #print(img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
