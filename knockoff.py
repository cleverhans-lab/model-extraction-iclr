import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models
import os.path as osp
import os
import time
from datetime import datetime
from collections import defaultdict as dd
import utils

import numpy as np


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=10, writer=None):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

#        if (batch_idx + 1) % log_interval == 0:
#         print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
#             exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
#             loss.item(), acc, correct, total))

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc

def test_step(model, test_loader, criterion, device, epoch=0., silent=True, writer=None, victimmodel = None):
    model.eval()
    test_loss = 0.
    correct = 0
    correct2 = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            #loss = criterion(outputs, targets)
            nclasses = outputs.size(1)

            #test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if victimmodel != None:
                outputs2 = victimmodel(inputs)
                _, predicted2 = outputs2.max(1)
                correct2 += predicted.eq(predicted2).sum().item()
                acc2 = 100. * correct2 / total

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
                                                                             correct, total))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
    if victimmodel == None:
        return test_loss, acc
    else:
        return test_loss, acc, acc2

def train_model(model, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, filerawacc=None, filerawacc2 = None, length = None, victimmodel = None, **kwargs):

    if device is None:
        device = torch.device('cuda')
    run_id = str(datetime.now())
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = testset
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.
    best_test_acc2 = -1
    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None and victimmodel == None:
            test_loss, test_acc = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)
        elif test_loader is not None:
            test_loss, test_acc, test_acc2 = test_step(model, test_loader, criterion_test,
                                            device, epoch=epoch, victimmodel=victimmodel)
            best_test_acc = max(best_test_acc, test_acc)
            best_test_acc2 = max(best_test_acc2, test_acc2)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)
    if filerawacc != None:
        utils.augmented_print(
            f'{length},{best_test_acc},knockoff,0',
            filerawacc,
            flush=True)
    if filerawacc2 != None:
        utils.augmented_print(
            f'{length},{best_test_acc2},knockoff,0',
            filerawacc2,
            flush=True)
    return model


# def train_adaptive(stolenmodel, victimmodel, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
#                 device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
#                 epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
#                 writer=None, filerawacc=None, filerawacc2 = None, length = None, **kwargs):
#     """Knockoff training with the adaptive strategy: This consists of an iterative training and querying process. """


# TODO: Update with code from ART library
#     nb_actions = 10 # 10 possible classes
#     y_avg = np.zeros(10)
#     reward = "all"
#     if reward == "all":
#             reward_avg = np.zeros(3)
#             reward_var = np.zeros(3)

#     h_func = np.zeros(nb_actions)
#     learning_rate = np.zeros(nb_actions)
#     probs = np.ones(nb_actions) / nb_actions
#     selected_x = []
#     queried_labels = []

#     avg_reward = 0.0

#     for iteration in range(1, epochs+1):
#             # Sample an action
#             action = np.random.choice(np.arange(0, nb_actions), p=probs)

#             # Sample data to attack
#             sampled_x = self._sample_data(x, y, action)
#             selected_x.append(sampled_x)

#             # Query the victim classifier
#             y_output = self.estimator.predict(x=np.array([sampled_x]), batch_size=self.batch_size_query)
#             fake_label = np.argmax(y_output, axis=1)
#             fake_label = to_categorical(labels=fake_label, nb_classes=self.estimator.nb_classes)
#             queried_labels.append(fake_label[0])

#             # Train the thieved classifier
#             thieved_classifier.fit(
#                 x=np.array([sampled_x]),
#                 y=fake_label,
#                 batch_size=self.batch_size_fit,
#                 nb_epochs=1,
#                 verbose=0,
#             )

#             # Test new labels
#             y_hat = stolenmodel.predict(x=np.array([sampled_x]), batch_size=self.batch_size_query)

#             # Compute rewards
#             reward = self._reward(y_output, y_hat, iteration)
#             avg_reward = avg_reward + (1.0 / iteration) * (reward - avg_reward)

#             # Update learning rate
#             learning_rate[action] += 1

#             # Update H function
#             for i_action in range(nb_actions):
#                 if i_action != action:
#                     h_func[i_action] = (
#                         h_func[i_action] - 1.0 / learning_rate[action] * (reward - avg_reward) * probs[i_action]
#                     )
#                 else:
#                     h_func[i_action] = h_func[i_action] + 1.0 / learning_rate[action] * (reward - avg_reward) * (
#                         1 - probs[i_action]
#                     )

#             # Update probs
#             aux_exp = np.exp(h_func)
#             probs = aux_exp / np.sum(aux_exp)

#         # Train the thieved classifier the final time
#         stolenmodel.fit(
#             x=np.array(selected_x),
#             y=np.array(queried_labels),
#             batch_size=self.batch_size_fit,
#             nb_epochs=self.nb_epochs,
#         )
#     return stolenmodel

