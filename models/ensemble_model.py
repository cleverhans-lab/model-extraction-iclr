import numpy as np
import os
import torch
from torch import nn as nn
import torch.nn.functional as F

import utils

from queryset import AGGREGATED_LABELS
from queryset import TARGETS
from models.utils_models import get_model_name_by_id


class EnsembleModel(nn.Module):
    """Noisy ensemble of private models.
    All the models for the ensemble are pre-cached in memory.
    """

    def __init__(self, model_id: int, private_models, args):
        """

        :param model_id: id of the model (-1 denotes all private models).
        :param private_models: list of private models
        :param args: program parameters
        """
        super(EnsembleModel, self).__init__()
        self.id = model_id
        if self.id == -1:
            self.name = f"ensemble(all)"
        else:
            # This is ensemble for private model_id.
            self.name = get_model_name_by_id(id=model_id)
        self.num_classes = args.num_classes
        print("Building ensemble model '{}'!".format(self.name))
        self.ensemble = private_models

    def __len__(self):
        return len(self.ensemble)

    def evaluate(self, evalloader, args):
        """Evaluate the accuracy of noisy ensemble model."""
        gap_list = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            for data, target in evalloader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                for model in self.ensemble:
                    output = model(data)
                    onehot = utils.one_hot(output.data.max(dim=1)[1].cpu(),
                                           self.num_classes)
                    votes += onehot
                # Add Gaussian noise
                assert args.sigma_gnmax >= 0
                if args.sigma_gnmax > 0:
                    noise = torch.from_numpy(
                        np.random.normal(0., args.sigma_gnmax, (
                            data.shape[0], self.num_classes))).float()
                    votes += noise
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()
                preds = votes.max(dim=1)[1].numpy().astype(np.int64)
                target = target.data.cpu().numpy().astype(np.int64)
                for label, pred, gap in zip(target, preds, gaps):
                    gap_list[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
        total = correct.sum() + wrong.sum()
        assert total == len(evalloader.dataset)
        return 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gap_list.sum() / total, gap_list / (
                       correct + wrong)

    def inference(self, unlabeled_dataloader, args):
        """Generate raw ensemble votes for RDP analysis_test."""
        all_votes = []
        end = 0
        correct_labels = 0
        total_labels = 0
        with torch.no_grad():
            for data, targets in unlabeled_dataloader:
                if args.cuda:
                    data = data.cuda()
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                if args.dataset == "imagenet":
                    if args.cuda:
                        model = self.ensemble.cuda()
                    else:
                        model = self.ensemble
                    output = model(data)
                    if args.vote_type == 'discrete':
                        label = output.argmax(dim=1).cpu()
                        model_votes = utils.one_hot(label, self.num_classes)
                    elif args.vote_type == 'probability':
                        model_votes = F.softmax(output, dim=1).cpu()
                    else:
                        label = output.argmax(dim=1).cpu()
                        model_votes = utils.one_hot(label, self.num_classes)
                    votes += model_votes
                else:
                    for model in self.ensemble:
                        output = model(data)
                        if args.vote_type == 'discrete':
                            label = output.argmax(dim=1).cpu()
                            correct_labels += (label == targets).int().sum()
                            total_labels += len(targets)
                            model_votes = utils.one_hot(label, self.num_classes)
                        elif args.vote_type == 'probability':
                            model_votes = F.softmax(output, dim=1).cpu()
                        else:
                            label = output.argmax(dim=1).cpu()
                            model_votes = utils.one_hot(label, self.num_classes)
                        votes += model_votes
                all_votes.append(votes.numpy())
        if total_labels > 0:
            accuracy = correct_labels / total_labels
            print('accuracy: ', accuracy)

        all_votes = np.concatenate(all_votes, axis=0)
        assert all_votes.shape == (
            len(unlabeled_dataloader.dataset), self.num_classes)
        if args.vote_type == 'discrete':
            assert np.all(all_votes.sum(axis=-1) == len(self.ensemble))
        # filename = '{}-raw-votes-(mode-{})'.format(self.name, args.mode) # Not saving right now
        # filepath = os.path.join(args.ensemble_model_path, filename)
        # np.save(filepath, all_votes)
        return all_votes

    def query(self, queryloader, args, indices_queried, targets=None):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        indices_answered = []
        all_preds = []
        all_labels = []
        gaps_detailed = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            begin = 0
            end = 0
            for data, target in queryloader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                num_samples = data.shape[0]
                end += num_samples
                # Generate raw ensemble votes
                votes = torch.zeros((num_samples, self.num_classes))
                for model in self.ensemble:
                    output = model(data)
                    if args.vote_type == 'discrete':
                        label = output.argmax(dim=1).cpu()
                        model_votes = utils.one_hot(label, self.num_classes)
                    elif args.vote_type == 'probability':
                        model_votes = F.softmax(output, dim=1).cpu()
                    else:
                        raise Exception(
                            f"Unknown args.votes_type: {args.votes_type}.")
                    votes += model_votes

                # Threshold mechanism
                if args.sigma_threshold > 0:
                    noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                       num_samples)
                    vote_counts = votes.data.max(dim=1)[0].numpy()
                    answered = (vote_counts + noise_threshold) > args.threshold
                    indices_answered.append(
                        indices_queried[begin:end][answered])
                else:
                    answered = [True for _ in range(num_samples)]
                    indices_answered.append(indices_queried[begin:end])

                # GNMax mechanism
                assert args.sigma_gnmax > 0
                noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    data.shape[0], self.num_classes))
                preds = \
                    (votes + torch.from_numpy(noise_gnmax).float()).max(dim=1)[
                        1].numpy().astype(np.int64)[answered]
                all_preds.append(preds)
                # Gap between the ensemble votes of the two most probable classes.
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()[
                    answered]
                # Target labels
                target = target.data.cpu().numpy().astype(np.int64)[answered]
                all_labels.append(target)
                assert len(target) == len(preds) == len(gaps)
                for label, pred, gap in zip(target, preds, gaps):
                    gaps_detailed[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
                begin += data.shape[0]
        indices_answered = np.concatenate(indices_answered, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        total = correct.sum() + wrong.sum()
        assert len(indices_answered) == len(all_preds) == len(
            all_labels) == total
        filename = utils.get_aggregated_labels_filename(
            args=args, name=self.name)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, all_preds)
        return indices_answered, 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gaps_detailed.sum() / total, gaps_detailed / (
                       correct + wrong)
