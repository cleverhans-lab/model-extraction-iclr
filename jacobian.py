import os

import scipy.stats
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from functools import partial
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from active_learning import compute_utility_scores_greedyj
from model_extraction.deepfool import compute_utility_scores_deepfoolj
from torch.nn import functional as F
delta = np.linalg.norm
import abc
import numpy
import pickle
import numpy.matlib
import time
import bisect
from torchvision import transforms
import random


linear = True # Set to False for default strategy
active = False # Set to False for random selection of generated points when using random. Otherwise active learning (gap) will be used to select these points.


def jaugment(model, dataloader, args):
    """Returns new queries to make based on the Jacobian based Data Augmentation strategy (JBDA) """
    lda = 0.1  # Hyperparameter (default based on paper is 0.1)
    newitems = []
    tempitems = []
    if linear == True and active == False:
        temp = []
        for i in range(len(dataloader)):
            temp.append(i)
        if len(temp) > args.adaptive_batch_size:
            selectedindices = random.sample(temp, args.adaptive_batch_size) # Set according to adaptive batch size.
        else:
            selectedindices = temp

    if active == True and linear == True and len(dataloader) > 1000:
        for data, label in dataloader: # for data in dataloader
            if args.cuda:
                data = data.cuda()
            if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                data = data.reshape((-1, 1, 28, 28))
            elif args.dataset == "imagenet":
                data = data.reshape((1, -1, 224, 224))
            else:
                data = data.reshape((1, -1, 32, 32))

            model.eval()
            jacob = torch.autograd.functional.jacobian(model, data)
            jacob = jacob.cpu()
            jacob = jacob[0]
            #jacob[i][j] is ith output and jth input.
            row = jacob[int(label)]
            b = np.sign(row)
            data = data.cpu()
            tempitems.append(data+torch.mul(b, lda))  #x'
        utility = compute_utility_scores_deepfoolj(model, tempitems, args)
        selectedindices = sorted(range(len(utility)), key = lambda sub: utility[sub])[:500] # Select 500 points

    i = 0
    for data, label in dataloader:
        if linear == True:
            # select 500 samples.
            if i in selectedindices:
                if args.cuda:
                    data = data.cuda()
                if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                    data = data.reshape((-1, 1, 28, 28))
                elif args.dataset == "imagenet":
                    data = data.reshape((1, -1, 224, 224))
                else:
                    data = data.reshape((1, -1, 32, 32))
                model.eval()
                jacob = jacobian(model, data)
                # jacob[i][j] is ith output and jth input.
                row = jacob[int(label)]
                b = np.sign(row)
                b = torch.from_numpy(b)
                data = data.cpu()
                newitems.append(data + torch.mul(b, lda))  # x'

        else:
            # For each data item, we generate the new augmented one.
            if args.cuda:
                data = data.cuda()
            if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                data = data.reshape((-1, 1, 28, 28))
            elif args.dataset == "imagenet":
                data = data.reshape((1, -1, 224, 224))
            else:
                data = data.reshape((1, -1, 32, 32))
            model.eval()
            jacob = torch.autograd.functional.jacobian(model, data)
            jacob = jacob.cpu()
            jacob = jacob[0]
            #jacob[i][j] is ith output and jth input.
            row = jacob[int(label)]
            b = np.sign(row) # Term to add to data
            data = data.cpu()
            newitems.append(data+torch.mul(b, lda))  #x'
        i += 1
    model.train()
    return newitems

def jaugment2(model, dataloader, args):
    """Returns augmented samples to be queries based on the JBDA-TR (Targeted) attack: https://openreview.net/pdf?id=LucJxySuJcE and https://arxiv.org/pdf/1805.02628.pdf (pg 4)"""
    lda = 0.1
    newitems = []
    tempitems = []
    #print(dataloader)
    if linear == True and len(dataloader) > 1000 and active==False:
        temp = []
        for i in range(len(dataloader)):
            temp.append(i)
        selectedindices = random.sample(temp, 500) # Set according to adaptive batch size for other modes.
    if active == True and linear == True and len(dataloader) > 1000:
        for data, label2 in dataloader:
            if args.cuda:
                data = data.cuda()
            if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                data = data.reshape((-1, 1, 28, 28))
            elif args.dataset == "imagenet":
                data = data.reshape((1, -1, 224, 224))
            else:
                data = data.reshape((1, -1, 32, 32))
            label = random.randint(0, 9)
            if label == label2 :
                label = (label + random.randint(1, 9)) % 10
            jacob = torch.autograd.functional.jacobian(model, data)
            jacob = jacob.cpu()
            jacob = jacob[0]
            row = jacob[label]
            b = np.sign(row)
            data = data.cpu()
            tempitems.append(data-torch.mul(b, lda))  #x'
        utility = compute_utility_scores_deepfoolj(model, tempitems, args)
        selectedindices = sorted(range(len(utility)), key = lambda sub: utility[sub])[:500] # Select 500 points
    i = 0
    for data, label2 in dataloader:
        if linear == True and len(dataloader) > 1000:
            # select 500 samples.
            if i in selectedindices:
                if args.cuda:
                    data = data.cuda()
                if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                    data = data.reshape((-1, 1, 28, 28))
                elif args.dataset == "imagenet":
                    data = data.reshape((1, -1, 224, 224))
                else:
                    data = data.reshape((1, -1, 32, 32))
                label = random.randint(0, 9)
                if label == label2: # Exclude the label equal to the actual output:
                    label = (label + random.randint(1,9)) % 10
                jacob = torch.autograd.functional.jacobian(model, data)
                jacob = jacob.cpu()
                jacob = jacob[0]
                row = jacob[label]
                b = np.sign(row)  # Term to add to input.
                data = data.cpu()
                newitems.append(data - torch.mul(b, lda))
        else:
            # For each data item, we generate the new augmented one.
            if args.cuda:
                data = data.cuda()
            if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                data = data.reshape((-1, 1, 28, 28))
            elif args.dataset == "imagenet":
                data = data.reshape((1, -1, 224, 224))
            else:
                data = data.reshape((1, -1, 32, 32))
            label = random.randint(0,9)
            if label == label2:
                label = (label + random.randint(1, 9)) % 10
            model.eval()
            jacob = torch.autograd.functional.jacobian(model, data)
            jacob = jacob.cpu()
            jacob = jacob[0]
            row = jacob[label]
            b = np.sign(row) 
            data = data.cpu()
            newitems.append(data-torch.mul(b, lda))
        i+=1
    model.train()
    return newitems



# Alternative functions from https://github.com/wanglouis49/pytorch-adversarial_box/blob/bddb5a899a7658182ea78063fd7ec405de083956/adversarialbox/attacks.py  :

def to_var(x, requires_grad=False, volatile=False):
    """
    Variable type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(x, requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives