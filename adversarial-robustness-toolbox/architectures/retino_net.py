import torch
import torch.nn as nn
from torch.nn import functional as F

# from https://github.com/YijinHuang/pytorch-DR/blob/reimplement/model.py

class RMSPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(RMSPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = torch.pow(x, 2)
        x = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        x = torch.sqrt(x)
        return x

class Conv2dUntiedBias(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, kernel_size, stride=1, padding=0):
        super(Conv2dUntiedBias, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels, height, width))

    def forward(self, x):
        output = F.conv2d(x, self.weight, None, self.stride, self.padding)
        output += self.bias.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        return output
        
class RetinoNet(nn.Module):
    def __init__(self, name, args, net_size='small', input_size=112, feature_dim=512):
        super(RetinoNet, self).__init__()
        
        self.name = name
        self.args = args
        # require inputs width and height in each layer because of the using of untied biases.
        sizes = self.cal_sizes(net_size, input_size)

        # named layers
        self.conv = nn.Sequential()
        if net_size in ['small', 'medium', 'large']:
            # 1-11 layers
            small_conv = nn.Sequential(
                self.basic_conv2d(3, 32, sizes[0], sizes[0], kernel_size=5, stride=2, padding=2),
                self.basic_conv2d(32, 32, sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(32, 64, sizes[1], sizes[1], kernel_size=5, stride=2, padding=2),
                self.basic_conv2d(64, 64, sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(64, 64, sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(64, 128, sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(128, 128, sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(128, 128, sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            )
            self.conv.add_module('small_conv', small_conv)

        if net_size in ['medium', 'large']:
            # 12-15 layers
            medium_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(128, 256, sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(256, 256, sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(256, 256, sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
            )
            self.conv.add_module('medium_conv', medium_conv)

        if net_size in ['large']:
            # 16-18 layers
            large_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(256, 512, sizes[4], sizes[4], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(512, 512, sizes[4], sizes[4], kernel_size=3, stride=1, padding=1),
            )
            self.conv.add_module('large_conv', large_conv)

        # RMSPooling layer
        self.conv.add_module('rmspool', RMSPool(3, 3))

        # regression part
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 1024),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 1)
        )

        # initial parameters
        for m in self.modules():
            if isinstance(m, Conv2dUntiedBias) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1)
                nn.init.constant_(m.bias, 0.05)

    def basic_conv2d(self, in_channels, out_channels, height, width, kernel_size, stride, padding):
        return nn.Sequential(
            Conv2dUntiedBias(in_channels, out_channels, height, width, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        features = self.conv(x)
        # reshape to satisify maxpool1d input shape requirement
        features = features.view(features.size(0), 1, -1)
        predict = self.fc(features)
        predict = torch.squeeze(predict)
        return predict

    # load part of pretrained_model like o_O solution \
    # using multi-scale image to train model by setting type to part \
    # or load full weights by setting type to full.
    def load_weights(self, pretrained_model_path, exclude=[]):
        pretrained_model = torch.load(pretrained_model_path)
        pretrained_dict = pretrained_model.state_dict()
        if isinstance(pretrained_model, nn.DataParallel):
            pretrained_dict = {key[7:]: value for key, value in pretrained_dict.items()}
        model_dict = self.state_dict()

        # exclude
        for name in list(pretrained_dict.keys()):
            # using untied biases will make it unable to reload.
            if name in model_dict.keys() and pretrained_dict[name].shape != model_dict[name].shape:
                pretrained_dict.pop(name)
                continue
            for e in exclude:
                if e in name:
                    pretrained_dict.pop(name)
                    break

        # load weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return pretrained_dict

    def layer_configs(self):
        model_dict = self.state_dict()
        return [(tensor, model_dict[tensor].size()) for tensor in model_dict]

    def cal_sizes(self, net_size, input_size):
        sizes = []
        if net_size in ['small', 'medium', 'large']:
            sizes.append(self._reduce_size(input_size, 5, 2, 2))
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 5, 2, 2))
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 3, 1, 1))
        if net_size in ['medium', 'large']:
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 3, 1, 1))
        if net_size in ['large']:
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 3, 1, 1))

        return sizes

    def _reduce_size(self, input_size, kernel_size, padding, stride):
        return (input_size + (2 * padding) - (kernel_size - 1) - 1) // stride + 1

# from https://www.kaggle.com/meenavyas/diabetic-retinopathy-detection

class SimpleRetinoNet(nn.Module):
    def __init__(self, name, args, input_channels=3, input_size=128):
        super(SimpleRetinoNet, self).__init__()
        self.name = name
        self.args = args
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.25)
        
        self.fc1 = nn.Linear(12544, 512)
        self.dropout4 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, self.args.num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.maxpool1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.maxpool2(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.maxpool3(x))
        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out
