from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import time

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn.functional as F

import analysis
import utils
from active_learning import compute_utility_scores_entropy
from active_learning import compute_utility_scores_entropyrev
from active_learning import compute_utility_scores_privacy, \
    compute_utility_scores_privacy2, compute_utility_scores_pate
from active_learning import compute_utility_scores_gap
from active_learning import compute_utility_scores_greedy
from active_learning import compute_utility_scores_random
from active_learning import PateKNN
from jacobian import jaugment, jaugment2
from datasets.dataset_custom_labels import DatasetLabels, DatasetProbs
from datasets.utils import show_dataset_stats
from model_extraction.deepfool import compute_utility_scores_deepfool
from models.ensemble_model import EnsembleModel
from models.load_models import load_private_models
from models.load_models import load_victim_model
from models.private_model import get_private_model_by_id
import torchvision.models as models
from utils import eval_distributed_model
from utils import from_result_to_str
from utils import metric
from utils import train_model
from utils import update_summary
from knockoff import train_model as trainknockoff
from knockoff import soft_cross_entropy
from architectures.resnet import ResNet18pre
import dfmenetwork


def get_utility_function(args):
    """
    Select the utility function.

    :param args: the arguments for the program.
    :return: the utility function (handler).
    """
    if args.mode == 'entropy':
        utility_function = compute_utility_scores_entropy
    elif args.mode == 'entropyrev':  # Reverse entropy method
        utility_function = compute_utility_scores_entropyrev
    elif args.mode == 'maxprivacy':  # maximize privacy cost
        utility_function = compute_utility_scores_privacy2
    elif args.mode == 'gap':
        utility_function = compute_utility_scores_gap
    elif args.mode == 'greedy':
        utility_function = compute_utility_scores_greedy
    elif args.mode == 'deepfool':
        utility_function = compute_utility_scores_deepfool
    elif args.mode == 'random':
        utility_function = compute_utility_scores_random
    elif args.mode == "knockoff":  # Knockoff Nets with Random querying
        utility_function = compute_utility_scores_random
    elif args.mode == "copycat":  # CopyCat CNN
        utility_function = compute_utility_scores_random
    elif args.mode == 'jacobian' or args.mode == 'jacobiantr':  # JBDA, JBDA-TR
        utility_function = compute_utility_scores_random
    elif args.mode == "inoutdist":  # Potential attack (combine ID and OOD Data)
        utility_function = compute_utility_scores_random
    elif args.mode == "worstcase":  # Attacker knows exact value of the privacy cost
        utility_function = compute_utility_scores_privacy
    elif args.mode == "worstcasepate":  # Attacker knows exact value of the pate cost
        utility_function = compute_utility_scores_pate
    else:
        raise Exception(f"Unknown query selection mode: {args.mode}.")
    return utility_function


def set_victim_model_path(args):
    if args.target_model == "victim":
        args.victim_model_path = os.path.join(
            args.path, 'private-models',
            args.dataset, args.architecture, '1-models')
    elif args.target_model == "pate":
        args.victim_model_path = os.path.join(
            args.path, 'private-models',
            args.dataset, args.architecture,
            '{}-models'.format(args.num_models))
        args.victim_model_path2 = os.path.join(
            args.path, 'private-models',
            args.dataset, args.architecture, '1-models')
    else:
        raise Exception(
            f"Target unspecified or unknown target type: {args.target_model}.")
    if os.path.exists(args.victim_model_path):
        print('args.victim_model_path: ', args.victim_model_path)
    else:
        raise Exception(
            "Victim Model does not exist at {}".format(args.victim_model_path))


def get_log_files(args, create_files=False):
    log_file_name = f"logs-num-epochs-{args.num_epochs}-{args.dataset}-{args.mode}-model-stealing.txt"
    log_file = os.path.join(args.path, log_file_name)
    file_raw_acc_name = f"log_raw_acc_PATE_cost_{args.mode}.txt"
    file_raw_acc = os.path.join(args.adaptive_model_path,
                                file_raw_acc_name)
    file_raw_acc2 = os.path.join(args.adaptive_model_path,
                                 f'log_raw_acc2_PATE_cost_{args.mode}.txt')
    file_raw_entropy_name = f"log_raw_entropy_{args.mode}.txt"
    file_raw_entropy2_name = f"log_raw_entropy2_{args.mode}.txt"
    file_raw_entropy = os.path.join(args.adaptive_model_path,
                                    file_raw_entropy_name)
    file_raw_entropy2 = os.path.join(args.adaptive_model_path,
                                     file_raw_entropy2_name)
    file_privacy_cost = os.path.join(args.adaptive_model_path,
                                     f'log_raw_pkNN_cost_{args.mode}.txt')
    file_raw_gap = os.path.join(args.adaptive_model_path,
                                f'log_raw_gap_{args.mode}.txt')
    file_raw_time = os.path.join(args.adaptive_model_path,
                                 f'log_raw_time_{args.mode}.txt')
    files = {
        'log_file': log_file,
        'file_raw_acc': file_raw_acc,
        'file_raw_acc2': file_raw_acc2,
        'file_raw_entropy': file_raw_entropy,
        'file_privacy_cost': file_privacy_cost,
        'file_raw_gap': file_raw_gap,
        'file_raw_entropy2': file_raw_entropy2,
        'file_raw_time': file_raw_time,
    }

    if create_files:
        for name in files:
            file_path = files[name]
            openfile = open(file_path, 'w+')
            openfile.close()

    return files


def close_log_files(files: dict):
    for file in files.values():
        try:
            file.close()
        except:
            pass


def print_initial_logs(args, evalloader=None):
    utils.augmented_print(
        "Training adaptive model on '{}' dataset!".format(args.dataset),
        args.log_file)
    utils.augmented_print(
        "Training adaptive model on '{}' architecture!".format(
            args.architecture), args.log_file)
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), args.log_file)
    utils.augmented_print(f"Initial learning rate: {args.lr}.", args.log_file)
    utils.augmented_print(
        "Number of epochs for training each model: {:d}".format(
            args.num_epochs), args.log_file)
    # Logs about the eval set
    if evalloader is not None:
        print(f'eval dataset: ', evalloader.dataset)
        show_dataset_stats(dataset=evalloader.dataset, args=args,
                           file=args.log_file,
                           dataset_name='eval')


def retrain(args, model, adaptive_loader, adaptive_dataset, evalloader,
            dp_eps, data_size, file_raw_acc, file_raw_acc2=None,
            victimmodel=None):
    summary = {
        'loss': [],
        'acc': [],
        'balanced_acc': [],
        'auc': [],
    }
    utils.augmented_print(
        f"Steps per epoch: {len(adaptive_loader)}.", args.log_file)
    if args.num_optimization_loop > 0:
        best_parameters = utils.bayesian_optimization_training_loop(
            args, model, adaptive_dataset, evalloader,
            patience=args.patience,
            num_optimization_loop=args.num_optimization_loop)

    else:
        model = get_private_model_by_id(args=args, id=0)
        best_parameters = {"lr": args.lr, "batch_size": args.batch_size}

    model = utils.train_with_bayesian_optimization_with_best_hyperparameter(
        args,
        model,
        adaptive_dataset,
        evalloader,
        parameters=best_parameters,
        patience=args.patience)

    # result = eval_distributed_model(
    #     model=model, dataloader=evalloader, args=args)
    result = utils.eval_model(args=args, model=model, dataloader=evalloader,
                              victimmodel=victimmodel)

    result_str = from_result_to_str(result=result, sep=' | ',
                                    inner_sep=': ')
    utils.augmented_print(text=result_str, file=args.log_file, flush=True)
    summary = update_summary(summary=summary, result=result)
    utils.augmented_print(
        f'{data_size},{result[metric.acc]},{args.mode},{dp_eps}',
        file_raw_acc,
        flush=True)
    if victimmodel != None:
        utils.augmented_print(
            f'{data_size},{result[metric.acc2]},{args.mode},{dp_eps}',
            file_raw_acc2,
            flush=True)
    utils.augmented_print(
        text=f'best hyperparameters : '
             f'lr {best_parameters["lr"]}, '
             f'batch size {best_parameters["batch_size"]}',
        file=args.log_file)

    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            utils.augmented_print(
                f"Average {key} of private models: {avg_value}", args.log_file)

    return result, model


def select_query_indices_based_on_utility(args, unlabeled_indices,
                                          unlabeled_dataset, utility_function,
                                          model,
                                          adaptive_batch_size):
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **args.kwargs)

    utility_scores = utility_function(
        model=model,
        dataloader=unlabeled_dataloader,
        args=args)

    # Sort unlabeled data according to their utility scores in
    # the descending order.
    all_indices_sorted = utility_scores.argsort()[::-1]
    # Take only the next adaptive batch size for labeling and this indices
    # that have not been labeled yet.
    selected_indices = []
    for index in all_indices_sorted:
        if index in unlabeled_indices:
            selected_indices.append(index)
            if len(selected_indices) == adaptive_batch_size:
                break
    return selected_indices


def get_victim_model_and_estimator(args):
    if args.dataset == "imagenet":
        cost_estimator_model = EnsembleModel(model_id=-1, args=args,
                                             private_models=models.resnet50(
                                                 pretrained=True))
    elif args.dataset == "cifar10" and args.architecture == "ResNet34":
        cost_estimator = []
        temp = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
        ckpt = 'dfmodels/teacher/cifar10-resnet34_8x.pt'
        temp.load_state_dict(torch.load(ckpt))
        if args.cuda:
            temp = temp.cuda()
        cost_estimator.append(temp)
        cost_estimator_model = EnsembleModel(model_id=-1, args=args,
                                             private_models=cost_estimator)
    elif args.dataset == "svhn" and args.architecture == "ResNet34":
        cost_estimator = []
        temp = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
        ckpt = 'dfmodels/teacher/svhn-resnet34_8x.pt'
        temp.load_state_dict(torch.load(ckpt))
        if args.cuda:
            temp = temp.cuda()
        cost_estimator.append(temp)
        cost_estimator_model = EnsembleModel(model_id=-1, args=args,
                                             private_models=cost_estimator)
    else:
        cost_estimator = load_private_models(args=args,
                                             model_path=args.private_model_path)
        #
        cost_estimator_model = EnsembleModel(model_id=-1, args=args,
                                             private_models=cost_estimator)

    if args.target_model == "victim":
        if args.dataset == "imagenet":
            victim_model = models.resnet50(pretrained=True)
            # victim_model = ResNet18pre()
            print("Loaded victim")
        elif args.dataset == "cifar10" and args.architecture == "ResNet34":
            victim_model = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
            ckpt = 'dfmodels/teacher/cifar10-resnet34_8x.pt'
            victim_model.load_state_dict(torch.load(ckpt))
            print("Loaded victim")
        elif args.dataset == "svhn" and args.architecture == "ResNet34":
            victim_model = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
            ckpt = 'dfmodels/teacher/svhn-resnet34_8x.pt'
            victim_model.load_state_dict(torch.load(ckpt))
            print("Loaded victim")
        else:
            victim_model = load_victim_model(args=args)
    elif args.target_model == "pate":
        # Create an ensemble model to be extracted / attacked.
        # We also load the single model which will be used for returning queries.
        private_models = load_private_models(args=args,
                                             model_path=args.private_model_path)
        victim_model = EnsembleModel(model_id=-1, args=args,
                                     private_models=private_models)
        if args.dataset == "imagenet":
            victim_model2 = models.resnet50(pretrained=True)
            print("Loaded victim2")
        elif args.dataset == "cifar10":
            victim_model2 = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
            ckpt = 'dfmodels/teacher/cifar10-resnet34_8x.pt'
            victim_model2.load_state_dict(torch.load(ckpt))
            print("Loaded victim2")
        elif args.dataset == "svhn":
            victim_model2 = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
            ckpt = 'dfmodels/teacher/svhn-resnet34_8x.pt'
            victim_model2.load_state_dict(torch.load(ckpt))
            print("Loaded victim2")
        else:
            temp = args.victim_model_path
            args.victim_model_path = args.victim_model_path2
            victim_model2 = load_victim_model(args=args)
            args.victim_model_path = temp
            print("Loaded victim2")
    else:
        raise Exception("target model not defined")

    if args.dataset in ["cifar10", "svhn"] and args.cuda and args.target_model == "pate":
        victim_model2 = victim_model2.cuda()
    if args.target_model == "pate":
        return victim_model, cost_estimator_model, victim_model2
    else:
        return victim_model, cost_estimator_model


def run_model_extraction(args, no_model_extraction=False):
    useserver = args.useserver
    start_time = time.time()

    files = get_log_files(args=args, create_files=True)
    log_file = files['log_file']
    file_raw_acc = files['file_raw_acc']
    file_raw_acc2 = files['file_raw_acc2']
    file_raw_entropy = files['file_raw_entropy']
    file_raw_entropy2 = files['file_raw_entropy2']
    file_privacy_cost = files['file_privacy_cost']
    file_raw_gap = files['file_raw_gap']
    file_raw_time = files['file_raw_time']
    utils.augmented_print(
        'queries,accuracy,type,privacy', file_raw_acc,
        flush=True)
    utils.augmented_print(
        'queries,accuracy,type,privacy', file_raw_acc2,
        flush=True)
    utils.augmented_print(
        '0,10,0,0', file_raw_acc,
        flush=True)
    utils.augmented_print(
        '0,10,0,0', file_raw_acc2,
        flush=True)
    utils.augmented_print(
        'queries,type,entropy', file_raw_entropy,
        flush=True)
    utils.augmented_print(
        '0,0,0', file_raw_entropy,
        flush=True)
    utils.augmented_print(
        'queries,type,gap', file_raw_gap,
        flush=True)
    utils.augmented_print(
        '0,0,0', file_raw_gap,
        flush=True)
    utils.augmented_print(
        'queries,type,pknn', file_privacy_cost,
        flush=True)
    utils.augmented_print(
        '0,0,0', file_privacy_cost,
        flush=True)
    evalloader = utils.load_evaluation_dataloader(args=args)

    args.log_file = log_file
    args.kwargs = utils.get_kwargs(args=args)
    args.save_model_path = args.adaptive_model_path

    set_victim_model_path(args=args)
    print_initial_logs(args=args, evalloader=evalloader)

    utility_function = get_utility_function(args=args)
    utility_function2 = compute_utility_scores_random
    # Values for entropyrev
    entrrev = {'cifar10': 10000, 'mnist': 4000, 'svhn': 8000}
    entropy_cost = 0
    entropy_cost2 = 0
    pate_cost = 0
    gap_cost = 0
    S = []  # Overall jacobian augmented set
    if args.mode == "jacobian" or args.mode == "jacobiantr":
        if args.dataset == "mnist" or args.dataset == "fashion-mnist":
            stolen_model = get_private_model_by_id(args=args, id=0)
        elif args.dataset == "imagenet":
            stolen_model = models.resnet18(pretrained=True)
        else:
            stolen_model = ResNet18pre()
            # stolen_model = get_private_model_by_id(args=args, id=0)
            print("Loaded stolen model")
    else:
        if args.architecture in ["ResNet34"]:  # stolen model will be resnet18 when victim is resnet34.
            from architectures.resnet import ResNet18
            stolen_model = ResNet18(name='model({:d})'.format(1), args=None)
        else:
            stolen_model = get_private_model_by_id(args=args, id=0)
        print("Loaded stolen model")
    # Adaptive model for training.
    if args.cuda:
        stolen_model = stolen_model.cuda()

    if args.target_model == "pate":
        victim_model, cost_estimator_model, victim_model2 = get_victim_model_and_estimator(args)
    else:
        victim_model, cost_estimator_model = get_victim_model_and_estimator(args)
    trainloader = None
    if args.target_model == 'victim':
        if args.dataset != "imagenet":  # 75% victim accuracy for imagenet.
            victim_acc = eval_distributed_model(
                model=victim_model, dataloader=evalloader, args=args)
            utils.augmented_print(
                text=f'accuracy of victim: {victim_acc[metric.acc]}.',
                file=log_file)
        else:
            if args.cuda:
                victim_model = victim_model.cuda()
        trainloader = utils.load_training_data(args=args)

        # initialize pknn
        pate_knn = PateKNN(model=victim_model, trainloader=trainloader,
                           args=args)

    # Prepare data.
    adaptive_batch_size = args.adaptive_batch_size

    # we are using a different dataset to steal this model
    if args.attacker_dataset:
        unlabeled_dataset = utils.get_attacker_dataset(
            args=args,
            dataset_name=args.attacker_dataset)
        print(f"attacker uses {args.attacker_dataset} dataset.")
    else:
        unlabeled_dataset = utils.get_unlabeled_set(args=args)

    if args.attacker_dataset == "tinyimages":
        unlabeled_dataset_cifar = utils.get_unlabeled_set(args=args)
        unlabeled_dataset = ConcatDataset(
            [unlabeled_dataset_cifar, unlabeled_dataset])

    total_data_size = len(unlabeled_dataset)

    if args.mode == "inoutdist":
        if args.attacker_dataset:
            unlabeled_dataset2 = utils.get_unlabeled_set(args=args)
        else:
            raise Exception(
                "Must use a seperate attacker dataset for mode inoutdist")
        total_data_size2 = len(unlabeled_dataset2)

    print(f"There are {total_data_size} unlabeled points in total.")

    # Initially all indices are unlabeled.
    unlabeled_indices = set([i for i in range(0, total_data_size)])
    # We will progressively add more labeled indices.
    labeled_indices = []
    # All labels extracted from the attacked model.
    all_labels = np.array([])
    # all probs extracted from the model (for knockoff).
    all_probs = None
    if args.mode == "inoutdist":
        unlabeled_indices2 = set([i for i in range(0, total_data_size2)])
        labeled_indices2 = []
        all_labels2 = np.array([])
    if args.attacker_dataset == "tinyimages":  # stop at specified limits for large datasets
        if args.mode in ["random", "entropy", "entropyrev", "gap",
                         "worstcase"]:
            data_iterator = range(
                adaptive_batch_size, 50001, adaptive_batch_size)
        elif args.mode == "jacobian":
            data_iterator = range(
                adaptive_batch_size, 150001, adaptive_batch_size)
        else:
            data_iterator = range(
                adaptive_batch_size, 10001, adaptive_batch_size)
    elif args.dataset == "imagenet":
        data_iterator = range(
            adaptive_batch_size, 50001, adaptive_batch_size)
    elif args.attacker_dataset == "imagenet":
        data_iterator = range(
            adaptive_batch_size, total_data_size + 1, adaptive_batch_size)
    else:
        data_iterator = range(
            adaptive_batch_size, total_data_size + 1, adaptive_batch_size)
    if args.mode == "inoutdist":
        data_iterator1 = range(
            adaptive_batch_size, total_data_size + 1, adaptive_batch_size)
        data_iterator2 = range(500, total_data_size2 + 1, 500)
        data_iterator = [val for pair in zip(data_iterator1, data_iterator2) for
                         val in pair]
    if args.mode == "jacobian":
        data_iterator = range(
            adaptive_batch_size, 76801, adaptive_batch_size)
    retrain_extracted_model = args.retrain_extracted_model  # save the parameter retrain_extracted_model
    for i, data_size in enumerate(data_iterator):
        # start1 = time.time()
        if no_model_extraction:
            # Extreme case for debugging: Since model is not being extracted, use victim's model for the selection of queries.
            if i % 2 == 0 or args.mode != "inoutdist":
                selected_indices = select_query_indices_based_on_utility(
                    args=args, unlabeled_indices=unlabeled_indices,
                    unlabeled_dataset=unlabeled_dataset,
                    utility_function=utility_function,
                    model=victim_model,
                    adaptive_batch_size=adaptive_batch_size)
            elif args.mode == "inoutdist":
                selected_indices = select_query_indices_based_on_utility(
                    args=args, unlabeled_indices=unlabeled_indices2,
                    unlabeled_dataset=unlabeled_dataset2,
                    utility_function=utility_function,
                    model=victim_model,
                    adaptive_batch_size=500)
        else:
            if i == 0:
                print("First iteration, no retraining")
                args.retrain_extracted_model = False

            else:
                args.retrain_extracted_model = retrain_extracted_model
            if args.mode in ["worstcase", "worstcasepate"]:
                # Assume the user has access to the actual scores computed on the victims end
                selected_indices = select_query_indices_based_on_utility(
                    args=args, unlabeled_indices=unlabeled_indices,
                    unlabeled_dataset=unlabeled_dataset,
                    utility_function=utility_function, model=victim_model,
                    adaptive_batch_size=adaptive_batch_size)
            elif i == 0 and args.mode == "jacobian":
                if args.dataset == "mnist":
                    selected_indices = select_query_indices_based_on_utility(
                        args=args, unlabeled_indices=unlabeled_indices,
                        unlabeled_dataset=unlabeled_dataset,
                        utility_function=utility_function, model=stolen_model,
                        adaptive_batch_size=150)
                else:
                    selected_indices = select_query_indices_based_on_utility(
                        args=args, unlabeled_indices=unlabeled_indices,
                        unlabeled_dataset=unlabeled_dataset,
                        utility_function=utility_function, model=stolen_model,
                        adaptive_batch_size=500)
            elif i % 2 == 0 or args.mode != "inoutdist":
                if data_size < entrrev[
                    args.dataset] and args.mode == "entropyrev" and args.attacker_dataset:
                    selected_indices = select_query_indices_based_on_utility(
                        args=args, unlabeled_indices=unlabeled_indices,
                        unlabeled_dataset=unlabeled_dataset,
                        utility_function=utility_function2, model=stolen_model,
                        adaptive_batch_size=adaptive_batch_size)
                else:
                    selected_indices = select_query_indices_based_on_utility(
                        args=args, unlabeled_indices=unlabeled_indices,
                        unlabeled_dataset=unlabeled_dataset,
                        utility_function=utility_function, model=stolen_model,
                        adaptive_batch_size=adaptive_batch_size)
            elif args.mode == "inoutdist":
                selected_indices = select_query_indices_based_on_utility(
                    args=args, unlabeled_indices=unlabeled_indices2,
                    unlabeled_dataset=unlabeled_dataset2,
                    utility_function=utility_function, model=stolen_model,
                    adaptive_batch_size=500)
        # Remove indices that we choose for this query.
        if i % 2 == 0 or args.mode != "inoutdist":
            if args.mode == "jacobian":
                pass
            else:
                unlabeled_indices = unlabeled_indices.difference(
                    selected_indices)
                assert len(unlabeled_indices) == total_data_size - data_size
        else:
            unlabeled_indices2 = unlabeled_indices2.difference(selected_indices)
        if args.mode in ["jacobian", "jacobiantr"]:
            if i == 0:
                unlabeled_subset = Subset(unlabeled_dataset,
                                          list(selected_indices))
                unlabeled_subloader = DataLoader(
                    unlabeled_subset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    **args.kwargs)
                S = []
                for data, _ in unlabeled_subset:
                    S.append(data)
            else:
                unlabeled_subset = []
                for i in range(len(Scur)):
                    if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                        unlabeled_subset.append((Scur[i].reshape(1, 28, 28),
                                                 0))  # placeholder value for the label as 0
                    elif args.dataset == "imagenet":
                        unlabeled_subset.append(
                            (Scur[i].reshape(3, 224, 224), 0))
                    else:
                        unlabeled_subset.append((Scur[i].reshape(3, 32, 32), 0))
                unlabeled_subloader = DataLoader(
                    unlabeled_subset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    **args.kwargs)
                S = S + Scur  # Full set S saved i.e. Sp
        else:
            if i % 2 == 0 or args.mode != "inoutdist":
                unlabeled_subset = Subset(unlabeled_dataset,
                                          list(selected_indices))
            else:
                unlabeled_subset = Subset(unlabeled_dataset2,
                                          list(selected_indices))
            unlabeled_subloader = DataLoader(
                unlabeled_subset,
                batch_size=args.batch_size,
                shuffle=False,
                **args.kwargs)

        new_labels = []
        new_probs = []
        start1 = time.time()
        if args.target_model == "victim":
            if useserver:  # Uses POW server client setup
                predicted_logits, time1 = utils.get_predictionnet(
                    args=args, model=victim_model,
                    unlabeled_dataloader=unlabeled_subloader)
            else:
                predicted_logits = utils.get_prediction(
                    args=args, model=victim_model,
                    unlabeled_dataloader=unlabeled_subloader)
            new_labels = predicted_logits.argmax(axis=1).cpu()
            if args.mode == "knockoff":
                new_probs = F.softmax(predicted_logits, dim=1).cpu().detach()
        elif args.target_model in ["pate"]:
            # victim_model is ensemble model while victim_model2 is the single model to return labels
            predicted_logits = utils.get_prediction(
                args=args, model=victim_model2,
                unlabeled_dataloader=unlabeled_subloader)
            new_labels = predicted_logits.argmax(axis=1).cpu()
            if args.mode == "knockoff":
                new_probs = F.softmax(predicted_logits, dim=1).cpu().detach()
        end1 = time.time()  # Time for querying
        if args.attacker_dataset and useserver:
            start1 = 0
            end1 = time1
        if i % 2 == 0 or args.mode != "inoutdist":
            all_labels = np.concatenate([all_labels, new_labels])
            labeled_indices += list(selected_indices)
            if args.mode not in ["jacobian", "jacobiantr"]:
                assert len(labeled_indices) == data_size
                assert len(
                    unlabeled_indices.union(
                        labeled_indices)) == total_data_size
        else:
            all_labels2 = np.concatenate([all_labels2, new_labels])
            labeled_indices2 += list(selected_indices)
            assert len(labeled_indices2) == data_size
            assert len(
                unlabeled_indices2.union(
                    labeled_indices2)) == total_data_size2
        if args.mode == "knockoff":
            if all_probs == None:
                all_probs = new_probs
            else:
                all_probs = torch.cat([all_probs, new_probs])

        if args.mode == "jacobian" or args.mode == "jacobiantr":
            adaptive_dataset = []
            for i in range(len(S)):
                if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                    adaptive_dataset.append(
                        (S[i].reshape(1, 28, 28), all_labels[i]))
                elif args.dataset == "imagenet":
                    adaptive_dataset.append(
                        (S[i].reshape(3, 224, 224), all_labels[i]))
                else:
                    adaptive_dataset.append(
                        (S[i].reshape(3, 32, 32), all_labels[i]))
            # adaptive_dataset will be the combined labeled items up to now

        else:
            adaptive_dataset = Subset(unlabeled_dataset, labeled_indices)
            if args.mode == "knockoff":
                adaptive_dataset = DatasetProbs(adaptive_dataset, all_probs)
                adaptive_dataset2 = DatasetLabels(adaptive_dataset, all_labels)
            else:
                adaptive_dataset = DatasetLabels(adaptive_dataset, all_labels)
            if args.mode == "inoutdist":
                adaptive_dataset2 = Subset(unlabeled_dataset2, labeled_indices2)
                adaptive_dataset2 = DatasetLabels(adaptive_dataset2,
                                                  all_labels2)
                adaptive_dataset = ConcatDataset(
                    [adaptive_dataset, adaptive_dataset2])
        adaptive_loader = DataLoader(
            adaptive_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **args.kwargs)
        if args.mode == "knockoff" and args.target_model == "pate":
            adaptive_loader2 = DataLoader(
                adaptive_dataset2,
                batch_size=args.batch_size,
                shuffle=False,
                **args.kwargs)
        k = 0
        if args.target_model == "victim":
            # record entropy and gap costs
            entropy_scores = compute_utility_scores_entropy(model=victim_model,
                                                            dataloader=unlabeled_subloader,
                                                            args=args)
            entropy_cost += entropy_scores.sum()
            gap_scores = compute_utility_scores_gap(
                model=victim_model,
                dataloader=unlabeled_subloader,
                args=args)
            gap_cost += gap_scores.sum()
            datalength = len(all_labels)
            if args.mode == "inoutdist":
                datalength += len(all_labels2)
            print("entropy cost")
            utils.augmented_print(
                f'{datalength},{args.mode},{entropy_cost}', file_raw_entropy,
                flush=True)
            print("gap cost")
            utils.augmented_print(
                f'{datalength},{args.mode},{gap_cost}', file_raw_gap,
                flush=True)
            if args.mode not in ["jacobian", "jacobiantr", "knockoff",
                                 "copycat"] and args.dataset != "imagenet":
                entropy_scores2 = compute_utility_scores_entropy(
                    model=stolen_model, dataloader=unlabeled_subloader,
                    args=args)
            if trainloader:
                # record pknn privacy cost
                pate_cost = pate_knn.compute_privacy_cost(
                    unlabeled_loader=unlabeled_subloader)
                utils.augmented_print(
                    f'{datalength},{args.mode},{pate_cost}', file_privacy_cost,
                    flush=True)
            if args.mode not in ["jacobian", "jacobiantr", "knockoff",
                                 "copycat"] and args.dataset != "imagenet":
                entropy_cost2 += entropy_scores2.sum()
                print("entropy 2 cost")
                utils.augmented_print(
                    f'{datalength},{args.mode},{entropy_cost2}',
                    file_raw_entropy2,
                    flush=True)
        if args.mode == "jacobian":
            # Scur is the new points generated by JBDA.
            start3 = time.time()
            if len(all_labels) < 150000:
                Scur = jaugment(stolen_model, adaptive_dataset, args)
            end3 = time.time()
        elif args.mode == "jacobiantr":
            start3 = time.time()
            Scur = jaugment2(stolen_model, adaptive_dataset, args)
            end3 = time.time()

        if args.target_model == "pate":
            entropy_scores = compute_utility_scores_entropy(model=victim_model2,
                                                            dataloader=unlabeled_subloader,
                                                            args=args)
            entropy_cost += entropy_scores.sum()
            gap_scores = compute_utility_scores_gap(
                model=victim_model2,
                dataloader=unlabeled_subloader,
                args=args)
            gap_cost += gap_scores.sum()
            datalength = len(all_labels)
            print("entropy cost")
            utils.augmented_print(
                f'{datalength},{args.mode},{entropy_cost}', file_raw_entropy,
                flush=True)
            print("gap cost")
            utils.augmented_print(
                f'{datalength},{args.mode},{gap_cost}', file_raw_gap,
                flush=True)

            if args.mode == "knockoff":
                votes_victim = victim_model.inference(adaptive_loader2, args)
            else:
                votes_victim = victim_model.inference(adaptive_loader, args)
            datalength = len(votes_victim)
            # Calculate PATE privacy cost
            pate_cost= 0
            for i in range(datalength):
                curvote = votes_victim[i][np.newaxis, :]
                max_num_query, dp_eps, partition, answered, order_opt = analysis.analyze_multiclass_confident_gnmax(
                    votes=curvote,
                    threshold=0,
                    sigma_threshold=0,
                    sigma_gnmax=args.sigma_gnmax,
                    budget=args.budget,
                    file=None,
                    delta=args.delta,
                    show_dp_budget=False,
                    args=args
                )
                pate_cost += dp_eps[0]
            print('pate cost')
            utils.augmented_print(
                f'{datalength},{args.mode},{pate_cost}', file_privacy_cost,
                flush=True)

        if args.mode == "jacobian":
            # Scur is the new points generated by JBDA.
            start3 = time.time()
            if len(all_labels) < 150000:
                Scur = jaugment(stolen_model, adaptive_dataset, args)
            end3 = time.time()
        elif args.mode == "jacobiantr":
            start3 = time.time()
            Scur = jaugment2(stolen_model, adaptive_dataset, args)
            end3 = time.time()

        # Logs about the adaptive train set.
        if args.debug is True:
            show_dataset_stats(dataset=adaptive_dataset,
                               args=args,
                               file=log_file,
                               dataset_name='private train')

        if no_model_extraction:
            # skip all the model saving/training logic
            if args.mode == "jacobian" or args.mode == "jacobiantr":
                print("time for querying", end1 - start1)
                print("Time for JBDA (next iter)", end3 - start3)
                utils.augmented_print(
                    f'{datalength},{args.mode},{end1 - start1},{end3 - start3}',
                    file_raw_time,
                    flush=True)
            else:
                print("time for querying", end1 - start1)
                utils.augmented_print(
                    f'{datalength},{args.mode},{end1 - start1}',
                    file_raw_time,
                    flush=True)
            continue
        start2 = time.time()
        if args.mode in ["jacobian", "jacobiantr", "copycat"] and args.target_model != "pate":
            # Also calculates fidelity accuracy
            result, model = retrain(args=args, model=stolen_model,
                                    file_raw_acc=file_raw_acc,
                                    evalloader=evalloader,
                                    adaptive_dataset=adaptive_dataset,
                                    adaptive_loader=adaptive_loader,
                                    dp_eps=pate_cost,
                                    data_size=len(all_labels),
                                    file_raw_acc2=file_raw_acc2,
                                    victimmodel=victim_model)
        elif args.mode == "knockoff" and args.target_model != "pate":
            if args.cuda:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            from torch import optim
            optimizer = optim.SGD(stolen_model.parameters(), 0.1)
            trainknockoff(stolen_model, adaptive_dataset,
                          out_path=args.adaptive_model_path, batch_size=64,
                          criterion_train=soft_cross_entropy,
                          criterion_test=soft_cross_entropy,
                          testset=evalloader,
                          device=device, num_workers=10, lr=args.lr,
                          momentum=args.momentum, lr_step=30, lr_gamma=0.1,
                          resume=None,
                          epochs=100, log_interval=100, weighted_loss=False,
                          checkpoint_suffix='', optimizer=optimizer,
                          scheduler=None,
                          writer=None, filerawacc=file_raw_acc, filerawacc2 = file_raw_acc2,
                          length=len(all_probs), victimmodel=victim_model)
        elif args.target_model == "pate":
            if args.mode != "knockoff":
                result, model = retrain(args=args, model=stolen_model,
                                        file_raw_acc=file_raw_acc,
                                        evalloader=evalloader,
                                        adaptive_dataset=adaptive_dataset,
                                        adaptive_loader=adaptive_loader,
                                        dp_eps=pate_cost,
                                        data_size=len(all_labels),
                                        file_raw_acc2=file_raw_acc2,
                                        victimmodel=victim_model2)
            else:
                trainknockoff(stolen_model, adaptive_dataset,
                              out_path=args.adaptive_model_path,
                              batch_size=64,
                              criterion_train=soft_cross_entropy,
                              criterion_test=soft_cross_entropy,
                              testset=evalloader,
                              device=device, num_workers=10, lr=args.lr,
                              momentum=args.momentum, lr_step=30,
                              lr_gamma=0.1,
                              resume=None,
                              epochs=100, log_interval=100,
                              weighted_loss=False,
                              checkpoint_suffix='', optimizer=optimizer,
                              scheduler=None,
                              writer=None, filerawacc=file_raw_acc,
                              filerawacc2=file_raw_acc2,
                              length=len(all_probs),
                              victimmodel=victim_model2)
        else:
            result, model = retrain(args=args, model=stolen_model,
                                    file_raw_acc=file_raw_acc,
                                    evalloader=evalloader,
                                    adaptive_dataset=adaptive_dataset,
                                    adaptive_loader=adaptive_loader,
                                    dp_eps=pate_cost,
                                    data_size=datalength)
        # save checkpoint and time
        end2 = time.time()
        if args.mode == "jacobian" or args.mode == "jacobiantr":
            print("time for querying", end1 - start1)
            print("time for training", end2 - start2)
            print("Time for JBDA (next iter)", end3 - start3)
            utils.augmented_print(
                f'{datalength},{args.mode},{end1 - start1},{end2 - start2},{end3 - start3}',
                file_raw_time,
                flush=True)
        else:
            print("time for querying", end1 - start1)
            print("time for training", end2 - start2)
            utils.augmented_print(
                f'{datalength},{args.mode},{end1 - start1},{end2 - start2}',
                file_raw_time,
                flush=True)
        if args.mode != "knockoff":
            state = result
            state['epoch'] = i
            state['state_dict'] = model.module.state_dict()
    # if args.target_model == "victim":
    #     victim_acc = eval_distributed_model(
    #         model=victim_model, dataloader=evalloader, args=args)
    #     utils.augmented_print(text=f'accuracy of victim: {victim_acc["acc"]}.',
    #                           file=log_file)
    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", log_file,
                          flush=True)
    assert len(unlabeled_indices) == 0
    close_log_files(files=files)
