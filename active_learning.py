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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from analysis.private_knn import PrivateKnn

delta = np.linalg.norm
import abc
import numpy
import pickle
import numpy.matlib
import time
import bisect
from analysis import analyze_multiclass_gnmax, analyze_multiclass_confident_gnmax
import utils

class SamplingMethod(object):

    @abc.abstractmethod
    def __init__(self):
        __metaclass__ = abc.ABCMeta

    def flatten_X(self, X):
        shape = X.shape
        flat_X = X
        if len(shape) > 2:
            flat_X = np.reshape(X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):

    def __init__(self, metric='euclidean'):
        super().__init__()
        self.name = 'kcenter'
        self.metric = metric
        self.min_distances = None
        self.already_selected = []

    def update_distances(self, features, cluster_centers, only_new=True,
                         reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          features: features (projection) from model
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = features[cluster_centers]
            dist = pairwise_distances(features.detach().numpy(),
                                      x.detach().numpy(), metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, pool, model, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          pool: tuple of (X, Y)
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            # Assumes that the transform function takes in original data and not
            # flattened data.
            print('Getting transformed features...')
            features = model.forward(pool[0].float())
            print('Calculating distances...')
            self.update_distances(features, already_selected, only_new=False,
                                  reset_dist=True)
        except Exception as e:
            print(f"error: {e}")
            print('Using flat_X as features.')
            self.update_distances(features, already_selected, only_new=True,
                                  reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(pool[0].shape[0]))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances(features, [ind], only_new=True,
                                  reset_dist=False)
            new_batch.append(ind)
        print(
            f"Maximum distance from cluster centers is {max(self.min_distances)}.")
        self.already_selected = already_selected

        return new_batch



def greedy_k_center(model, pool, already_selected, batch_size):
    # note pool should have all points in a tuple of (X, Y)
    # already selected are the indices
    # this returns the indices o the selected samples
    selecter = kCenterGreedy()
    return selecter.select_batch_(pool, model, already_selected, batch_size)


def robust_k_center(x, y, z):
    budget = 10000

    start = time.clock()
    num_images = x.shape[0]
    dist_mat = numpy.matmul(x, x.transpose())

    sq = numpy.array(dist_mat.diagonal()).reshape(num_images, 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()

    elapsed = time.clock() - start
    print(f"Time spent in (distance computation) is: {elapsed}")

    num_images = 50000

    # We need to get k centers start with greedy solution
    budget = 10000
    subset = [i for i in range(1)]

    ub = UB
    lb = ub / 2.0
    max_dist = ub

    _x, _y = numpy.where(dist_mat <= max_dist)
    _d = dist_mat[_x, _y]
    subset = [i for i in range(1)]
    model = solve_fac_loc(_x, _y, subset, num_images, budget)
    # model.setParam( 'OutputFlag', False )
    x, y, z = model.__data
    delta = 1e-7
    while ub - lb > delta:
        print("State", ub, lb)
        cur_r = (ub + lb) / 2.0
        viol = numpy.where(_d > cur_r)
        new_max_d = numpy.min(_d[_d >= cur_r])
        new_min_d = numpy.max(_d[_d <= cur_r])
        print("If it succeeds, new max is:", new_max_d, new_min_d)
        for v in viol[0]:
            x[_x[v], _y[v]].UB = 0

        model.update()
        r = model.optimize()
        if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
            failed = True
            print("Infeasible")
        elif sum([z[i].X for i in range(len(z))]) > 0:
            failed = True
            print("Failed")
        else:
            failed = False
        if failed:
            lb = max(cur_r, new_max_d)
            # failed so put edges back
            for v in viol[0]:
                x[_x[v], _y[v]].UB = 1
        else:
            print("sol found", cur_r, lb, ub)
            ub = min(cur_r, new_min_d)
            model.write("s_{}_solution_{}.sol".format(budget, cur_r))


def get_model_name(model):
    if getattr(model, 'module', '') == '':
        return model.name
    else:
        return model.module.name


def compute_utility_scores_entropy(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    with torch.no_grad():
        # Entropy value as a proxy for utility.
        entropy = []
        for data, _ in dataloader:
            if args.cuda:
                data = data.cuda()
            output = model(data)
            prob = F.softmax(output, dim=1).cpu().numpy()
            entropy.append(scipy.stats.entropy(prob, axis=1))
        entropy = np.concatenate(entropy, axis=0)
        # Maximum entropy is achieved when the distribution is uniform.
        entropy_max = np.log(args.num_classes)
        # Sanity checks
        try:
            assert len(entropy.shape) == 1 and entropy.shape[0] == len(
                dataloader.dataset)
            assert np.all(entropy <= entropy_max) and np.all(0 <= entropy)
        except AssertionError:
            # change nan to 0 and try again
            entropy[np.isnan(entropy)] = 0
            assert len(entropy.shape) == 1 and entropy.shape[0] == len(
                dataloader.dataset)
            assert np.all(entropy <= entropy_max) and np.all(0 <= entropy)
            print("There are NaNs in the utlity scores, reset to 0")
        # Normalize utility scores to [0, 1]
        utility = entropy / entropy_max
        return utility

def compute_utility_scores_entropyrev(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset. Selects items with the smallest entropy first."""
    with torch.no_grad():
        # Entropy value as a proxy for utility.
        entropy = []
        for data, _ in dataloader:
            if args.cuda:
                data = data.cuda()
            output = model(data)
            prob = F.softmax(output, dim=1).cpu().numpy()
            entropy.append(scipy.stats.entropy(prob, axis=1))
        entropy = np.concatenate(entropy, axis=0)
        # Maximum entropy is achieved when the distribution is uniform.
        entropy_max = np.log(args.num_classes)
        # Sanity checks
        try:
            assert len(entropy.shape) == 1 and entropy.shape[0] == len(
                dataloader.dataset)
            assert np.all(entropy <= entropy_max) and np.all(0 <= entropy)
        except AssertionError:
            # change nan to 0 and try again
            entropy[np.isnan(entropy)] = 0
            assert len(entropy.shape) == 1 and entropy.shape[0] == len(
                dataloader.dataset)
            assert np.all(entropy <= entropy_max) and np.all(0 <= entropy)
            print("There are NaNs in the utlity scores, reset to 0")
        # Normalize utility scores to [0, 1]
        utility = entropy / entropy_max
        utility = -utility
        return utility

def compute_utility_scores_privacy(model, dataloader, args):
    trainloader = utils.load_training_data(args=args)
    pate_knn = PateKNN(model=model, trainloader=trainloader,
                       args=args)
    """Assign a utility score to each data sample from the unlabeled dataset. Selects items with the minimum privacy first."""
    with torch.no_grad():
        # Privacy value as a proxy for utility.
        privacy = []
        curcost = 0
        for data, target in dataloader:
            for j in range(len(data)):
                tempdataset = [(data[j], target[j])]
                tempcost = pate_knn.compute_privacy_cost(unlabeled_loader=DataLoader(
                tempdataset,
                batch_size=1,
                shuffle=False,
                ))
                privacy.append(tempcost-curcost)
                curcost = tempcost
        privacy = np.array(privacy)
        privacy_max = np.max(privacy)
        # Sanity checks
        # Normalize utility scores to [0, 1]
        utility = privacy / privacy_max
        utility = -utility
        return utility

def compute_utility_scores_pate(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset. Selects items with the minimum privacy first."""
    victim_model = model
    votes_victim = victim_model.inference(dataloader, args)
    datalength = len(votes_victim)
    privacy = []
    for i in range(datalength):
        curvote = votes_victim[i][np.newaxis, :]
        max_num_query, dp_eps, partition, answered, order_opt = analyze_multiclass_confident_gnmax(
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
        privacy.append(dp_eps[0])

    privacy = np.array(privacy)
    privacy_max = np.max(privacy)
    # Sanity checks
    # Normalize utility scores to [0, 1]
    utility = privacy / privacy_max
    utility = -utility
    return utility

def compute_utility_scores_privacy2(model, dataloader, args):
    trainloader = utils.load_training_data(args=args)
    pate_knn = PateKNN(model=model, trainloader=trainloader,
                       args=args)
    """Assign a utility score to each data sample from the unlabeled dataset. Selects items with the maximum privacy first."""
    with torch.no_grad():
        # Privacy value as a proxy for utility.
        privacy = []
        curcost = 0
        for data, target in dataloader:
            for j in range(len(data)):
                tempdataset = [(data[j], target[j])]
                tempcost = pate_knn.compute_privacy_cost(unlabeled_loader=DataLoader(
                tempdataset,
                batch_size=1,
                shuffle=False,
                ))
                privacy.append(tempcost-curcost)
                curcost = tempcost
        print("sorted", privacy.sort())
        print("len", len(privacy))
        privacy = np.array(privacy)
        privacy_max = np.max(privacy)
        # Sanity checks
        # Normalize utility scores to [0, 1]
        utility = privacy / privacy_max
        return utility

def get_train_representations(model, trainloader, args):
    """
    Compute the train representations for the training set.
    :param model: ML model
    :param trainloader: data loader for training set
    :param args: the parameters for the program
    :return: training representations and their targets
    """
    train_represent = []
    train_labels = []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(trainloader):
            if args.cuda:
                data = data.cuda()
            outputs = model(data)
            outputs = F.log_softmax(outputs, dim=-1)
            outputs = outputs.cpu().numpy()
            train_represent.append(outputs)
            train_labels.append(target.cpu().numpy())
    train_represent = np.concatenate(train_represent, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    return train_represent, train_labels

def get_votes_for_pate_knn(model, unlabeled_loader, train_represent,
                           train_labels, args):
    """
    :param model: the model to be used
    :param unlabeled_loader: data points to be labeled - for which we compute
        the score
    :param train_represent: last layer representation for the teachers
    :param train_labels: labels for the teachers
    :param args: the program parameters
    :return: votes for each data point
    """

    # num_teachers: number of k nearest neighbors acting as teachers
    num_teachers = args.num_teachers_private_knn

    with torch.no_grad():
        # Privacy cost as a proxy for utility.
        votes = []
        targets = []
        predictions = []
        for data, target in unlabeled_loader:
            if args.cuda:
                data = data.cuda()
            outputs = model(data)
            outputs = F.log_softmax(outputs, dim=-1)
            outputs = outputs.cpu().numpy()
            targets.append(target.cpu().numpy())
            predictions.append(np.argmax(outputs, axis=-1))
            for output in outputs:
                dis = np.linalg.norm(train_represent - output, axis=-1)
                k_index = np.argpartition(dis, kth=num_teachers)[:num_teachers]
                teachers_preds = np.array(train_labels[k_index], dtype=np.int32)
                label_count = np.bincount(
                    teachers_preds, minlength=args.num_classes)
                votes.append(label_count)
    votes = np.stack(votes)
    sorted_votes = np.flip(np.sort(votes, axis=1), axis=1)
    gaps = (sorted_votes[:, 0] - sorted_votes[:, 1])
    return votes


def compute_utility_scores_pate_knn(
        model, unlabeled_loader, args, trainloader, train_represent=None,
        train_labels=None):
    """Assign a utility score to each data sample from the unlabeled dataset.
    Either trainloader or train_represent has to be provided.
    :param model: the model to be used
    :param unlabeled_loader: data points to be labeled - for which we compute
        the score
    :param args: the program parameters
    :param trainloader: the data loader for the training set
    :param train_represent: last layer representation for the teachers
    :param train_labels: labels for the teachers
    :return: utility score based on the privacy budget for each point in the
    dataset unlabeled_loader
    """
    if train_represent is None:
        assert trainloader is not None
        train_represent, train_labels = get_train_representations(
            model=model, trainloader=trainloader, args=args)

    votes = get_votes_for_pate_knn(
        model=model, train_labels=train_labels, train_represent=train_represent,
        args=args, unlabeled_loader=unlabeled_loader
    )

    max_num_query, dp_eps, _, _, _ = analyze_multiclass_gnmax(
        votes=votes,
        threshold=0,
        sigma_threshold=0,
        sigma_gnmax=args.sigma_gnmax_private_knn,
        budget=np.inf,
        delta=args.delta,
        show_dp_budget=args.show_dp_budget,
        args=args)
    # Make sure we compute the privacy loss for all queries.
    assert max_num_query == len(votes)
    privacy_cost = dp_eps
    return privacy_cost


class PateKNN:
    """
    Compute the privacy cost.
    """

    def __init__(self, model, trainloader, args):
        """
        Args:
            model: the victim model.
            trainloader: the data loader for the training data.
            args: the program parameters.
        """
        self.model = model
        self.args = args

        # Extract the last layer representation of the training points and their
        # ground-truth labels.
        self.train_represent, self.train_labels = get_train_representations(
            model=model, trainloader=trainloader, args=args)

        self.private_knn = PrivateKnn(
            delta=args.delta, sigma_gnmax=args.sigma_gnmax_private_knn,
            apply_data_independent_bound=args.apply_data_independent_bound)

    def compute_privacy_cost(self, unlabeled_loader):
        """
        Args:
            unlabeled_loader: data loader for new queries.
        Returns:
            The total privacy cost incurred by all the queries seen so far.
        """
        votes = get_votes_for_pate_knn(
            model=self.model, train_labels=self.train_labels,
            train_represent=self.train_represent, args=self.args,
            unlabeled_loader=unlabeled_loader
        )

        dp_eps = self.private_knn.add_privacy_cost(votes=votes)

        return dp_eps

def compute_utility_scores_gap(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    with torch.no_grad():
        # Gap between the probabilities of the two most probable classes as a proxy for utility.
        gap = []
        for data, _ in dataloader:
            if args.cuda:
                data = data.cuda()
            output = model(data)
            sorted_output = output.sort(dim=-1, descending=True)[0]
            prob = F.softmax(sorted_output[:, :2], dim=1).cpu().numpy()
            gap.append(prob[:, 0] - prob[:, 1])
        gap = np.concatenate(gap, axis=0)
        # Sanity checks
        try:
            assert len(gap.shape) == 1 and gap.shape[0] == len(
                dataloader.dataset)
            assert np.all(gap <= 1) and np.all(
                0 <= gap), f"gaps: {gap.tolist()}"
        except AssertionError:
            # change nan to 0 and try again
            gap[np.isnan(gap)] = 0
            assert len(gap.shape) == 1 and gap.shape[0] == len(
                dataloader.dataset)
            assert np.all(gap <= 1) and np.all(
                0 <= gap), f"gaps: {gap.tolist()}"
            print("There are NaNs in the utlity scores, reset to 0")
        # Convert gap values into utility scores
        utility = 1 - gap
        return utility


def compute_utility_scores_greedy(model, dataloader, args):
    model.cpu()
    with torch.no_grad():
        samples = []
        for data, _ in dataloader:
            data = Variable(data)
            samples.append(data)
        samples = torch.cat(samples, dim=0)
        indices = greedy_k_center(model, (samples, None), [],
                                  len(dataloader.dataset))
        try:
            assert len(indices) == len(dataloader.dataset) and len(
                set(indices)) == len(dataloader.dataset)
        except AssertionError:
            print("Assertion Error In Greedy, return all zero utility scores")
            return np.zeros(len(dataloader.dataset))
        indices = np.array(indices)
        utility = np.zeros(len(dataloader.dataset))
        for i in range(len(indices)):
            utility[indices[i]] = (len(dataloader.dataset) - i) / float(
                len(dataloader.dataset))
        if args.cuda:
            model.cuda()
        return utility

def compute_utility_scores_greedyj(model, dataloader, args):
    model.cpu()
    with torch.no_grad():
        samples = []
        for data in dataloader:
            data = Variable(data)
            samples.append(data)
        samples = torch.cat(samples, dim=0)
        print("SAMPLES", len(samples))
        indices = greedy_k_center(model, (samples, None), [],
                                  len(dataloader))
        try:
            print("Indices", len(indices))
            print("Dataloader", len(dataloader))
            print("Set", len(set(indices)))
            print("Set2", len(set(dataloader)))
            assert len(indices) == len(dataloader) and len(  # PROBLEMS HERE
                set(indices)) == len(set(dataloader))
        except AssertionError:
            print("Assertion Error In Greedy, return all zero utility scores")
            return np.zeros(len(dataloader))
        indices = np.array(indices)
        utility = np.zeros(len(dataloader))
        for i in range(len(indices)):
            utility[indices[i]] = (len(dataloader) - i) / float(
                len(dataloader))
        if args.cuda:
            model.cuda()
        return utility


def compute_utility_scores_random(model, dataloader, args):
    return np.random.random(len(dataloader.dataset))
