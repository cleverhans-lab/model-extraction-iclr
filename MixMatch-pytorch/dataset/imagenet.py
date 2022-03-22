import numpy as np
from PIL import Image

import torchvision
from torchvision import transforms
import torch

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

# Here the victim has the cifar dataset while the attacker uses imagenet. i.e. labeled and unlabeled sets come from imagenet (compressed down to 32x32)
tempd = []
def get_cifar10(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = [  # May need to fix this!
        transforms.Resize(32),
        transforms.CenterCrop(32),
        #RandomPadandCrop(32),
        #RandomFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    test_dataset2 = torchvision.datasets.ImageNet(root = "/scratch/ssd002/datasets/imagenet256/", split='val',
                     transform=transforms.Compose(preprocessing))  # Only used here and for the later class to get the actual images.
    trainloader = torch.utils.data.DataLoader(test_dataset2, batch_size=50000)
    global tempd
    #tempd = []  # needs to be a numpy array
    for batch_id, (data, target) in enumerate(trainloader):
        tempd = data
        break
    # tempd = tempd.cpu().detach().numpy()
    # print(type(tempd))
    # print(tempd.shape) # Add if needed
    #print(tempd)
    #tempd = np.array(tempd)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(test_dataset2.targets, int(n_labeled/10)) # was base_dataset.target

    train_labeled_dataset = imagenet_labeledmod(root, train_labeled_idxs, train=False, transform=transform_train) # train=True
    train_unlabeled_dataset = imagenet_unlabeled(root, train_unlabeled_idxs, train=False, transform=TransformTwice(transform_train))
    val_dataset = imagenet_labeledmod(root, val_idxs, train=False, transform=transform_train, download=True)  # possibly change train = False here and above. was transform_test
    test_dataset = CIFAR10_labeled('./data', val_idxs, train=True, transform=transform_val, download=True) # change Train to False and use different indices.
    #print("Test shape", train_labeled_dataset[0][0].shape)
    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

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

    # Random selection for point:
    n_labeled = n_labeled_per_class * 10
    idxs = np.where(labels > -10)[0] # All points
    np.random.shuffle(idxs)
    train_labeled_idxs.extend(idxs[:n_labeled])
    train_unlabeled_idxs.extend(idxs[n_labeled: -1000])   #-500 here and below originally
    val_idxs.extend(idxs[-1000:])
    ent = 0
    gap = 0
    temp1 = np.load("imagenetent.npy")
    temp2 = np.load("imagenetgap.npy")
    for i in train_labeled_idxs:
        ent += temp1[i]
        gap += temp2[i]
    #pknn = 0
    total = n_labeled_per_class*10
    file = f"imagenet@{total}new/stats.txt"
    f = open(file, "w")
    f.write("Entropy: " + str(ent) + "\n")
    f.write("Gap: " + str(gap) + "\n")
    f.close()
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def normalise(x, mean=imagenet_mean, std=imagenet_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
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

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        #print(img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class imagenet_labeled(torchvision.datasets.ImageNet):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(imagenet_labeled, self).__init__(root, split = "val",
                 transform=transform, target_transform=target_transform)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        #print(img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
class imagenet_labeledmod(torchvision.datasets.ImageNet):
# Need to change a lot. Try using dataloaders or something. Just loop through to get self.data and self.targets
    def __init__(self, root, indexs=None, train=False,
                 transform=None, target_transform=None,
                 download=False):
        super(imagenet_labeledmod, self).__init__(root, split='val',
                 transform=transform, target_transform=target_transform)
        global tempd
        # trainloader = torch.utils.data.DataLoader(self, batch_size=1)
        # tempd = [] # needs to be a numpy array
        # for batch_id, (data, target) in enumerate(trainloader):
        #     tempd.append(data)
        #self.data = np.array(tempd)
        self.data = tempd
        #self.data = torch.tensor([1,2,3])
        if indexs is not None:
            #self.data = self.data[indexs]
            #self.data = tempd[indexs]
            #victim = load_private_model_by_id()
            # temp = []
            # for i in indexs:
            #self.targets = victim(self.data)
            # Use model predictions here?
            targets = np.load("imagenettargets.npy")
            self.targets = np.array(targets)#[indexs]
            #print(self.targets)
        #self.data = transpose(normalise(self.data))
        #self.data = transpose(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        #print(img.shape)
        if self.transform is not None: #Not needed because of the transforms being done already.
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class imagenet_unlabeled(imagenet_labeledmod):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(imagenet_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for i in range(len(self.targets))])


        