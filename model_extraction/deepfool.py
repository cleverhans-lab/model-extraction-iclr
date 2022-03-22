"""
Pytorch implementation of deepfool attack
deepfool function modified from https://github.com/Dawn-David/DeepFool_MNIST/blob/master/deepfool_fashion.py
"""
import copy
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from utils import get_device


def deepfool(image, net, device, gt_label=None, num_classes=10, overshoot=0.02,
             max_iter=50):
    """
       :param image: 1x1x28x28
       :param net: network
       :device: cuda or cpu
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    out = net(image)

    f_image = out.detach().cpu().numpy().flatten()

    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    if gt_label is not None and label != gt_label.tolist():
        return 0, 0, label, 0, image

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    pert_image = pert_image.reshape(image[0].shape)
    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig  # wk
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()  # fk

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())  # l'

            # determine which w_k to use

            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).to(
            device)

        x = Variable(pert_image, requires_grad=True)
        fs = net(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image


def get_model_name(model):
    if getattr(model, 'module', '') == '':
        return model.name
    else:
        return model.module.name


def compute_utility_scores_deepfool(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    # with torch.no_grad():
    # Entropy value as a proxy for utility.
    l2_deepfool = []
    device = get_device(args)[0]
    model = model.to(device)
    print('number of data samples: ', len(dataloader.dataset))
    for data, _ in dataloader:
        data = data.to(device)
        for image_ind in range(data.shape[0]):
            # go through each data point in batch
            image = data[image_ind].unsqueeze(0)
            # gt_label = targets[image_ind]
            r_tot, loop_i, label, _, _ = deepfool(
                image, model, device, None, max_iter=20,
                num_classes=10)
            l2_deepfool.append(np.linalg.norm(r_tot))
    l2_deepfool = np.array(l2_deepfool)
    # Sanity checks
    assert len(l2_deepfool.shape) == 1 and l2_deepfool.shape[0] == len(
        dataloader.dataset)
    # Normalize utility scores to [0, 1]
    utility = 1 - (l2_deepfool / max(l2_deepfool))
    try:
        assert (utility.max()) <= 1 and (utility.min()) >= 0
    except AssertionError:
        print(utility.max(), utility.min())
    # Save utility scores
    filename = "{}-utility-scores-(mode:deepfool)".format(
        get_model_name(model))
    if os.name == "nt":
        filename = "{}-utility-scores-(mode_deepfool)".format(
            get_model_name(model))
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, utility)
    return utility

def compute_utility_scores_deepfoolj(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    # with torch.no_grad():
    # Entropy value as a proxy for utility.
    l2_deepfool = []
    device = get_device(args)[0]
    model = model.to(device)
    print('number of data samples: ', len(dataloader))
    for data in dataloader:
        data = data.to(device)
        for image_ind in range(data.shape[0]):
            # go through each data point in batch
            image = data[image_ind].unsqueeze(0)
            # gt_label = targets[image_ind]
            r_tot, loop_i, label, _, _ = deepfool(
                image, model, device, None, max_iter=20,
                num_classes=10)
            l2_deepfool.append(np.linalg.norm(r_tot))
    l2_deepfool = np.array(l2_deepfool)
    # Sanity checks
    assert len(l2_deepfool.shape) == 1 and l2_deepfool.shape[0] == len(
        dataloader)
    # Normalize utility scores to [0, 1]
    utility = 1 - (l2_deepfool / max(l2_deepfool))
    try:
        assert (utility.max()) <= 1 and (utility.min()) >= 0
    except AssertionError:
        print(utility.max(), utility.min())
    # Save utility scores
    return utility