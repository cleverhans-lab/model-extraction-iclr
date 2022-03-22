'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
#import torchvision.models as models
import pretrainedmodels


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, name=''):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.name = name
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetpre(nn.Module): # Pretrained Resnet (only for resnet18) Based on https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5
    def __init__(self, block, num_blocks, num_classes=10, name=''):
        super(ResNetpre, self).__init__()
        #self.model = models.resnet18(pretrained=True)
        self.model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        self.classifier_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        batch_size, _, _, _= x.shape
        out = self.model.features(x)
        out = F.avg_pool2d(out, 1).reshape(batch_size, -1)
        out = self.classifier_layer(out)
        return out


def ResNet10(name, args):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=args.num_classes,
                  name=name)


def ResNet12(name, args):
    return ResNet(BasicBlock, [2, 1, 1, 1], num_classes=args.num_classes,
                  name=name)


def ResNet14(name, args):
    return ResNet(BasicBlock, [2, 2, 1, 1], num_classes=args.num_classes,
                  name=name)


def ResNet16(name, args):
    return ResNet(BasicBlock, [2, 2, 2, 1], num_classes=args.num_classes,
                  name=name)


# def ResNet18(name, args):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes,
#                   name=name)

def ResNet18(name, args = None): # Allows for model to be made without args.
    if args == None:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10,
                      name=name)
    else:
        # if args.dataset != "mnist" and args.mode in ["jacobian", "jacobiantr"]:
        #     return ResNetpre(BasicBlock, [2,2,2,2], num_classes=args.num_classes, name=name)
        # else:
        #     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes,
        #               name=name)
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes,
                      name=name)

def ResNet18pre():
    return ResNetpre(BasicBlock, [2,2,2,2], num_classes=10, name="1")

def ResNet34(name, args=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10, name=name)

# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def trial():
    from parameters import get_parameters
    args = get_parameters()
    args.num_classes = 10
    net = ResNet10(name='ResNet10', args=args)
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print('y size: ', y.size())


if __name__ == "__main__":
    trial()
