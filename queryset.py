from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from models.utils_models import get_model_name_by_id

INFO = "-{}-{}-labels-(mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f})-transfer-{}.npy"

RAW_QUERIES = "raw-queries" + INFO
QUERIES = "queries" + INFO
AGGREGATED_LABELS = "aggregated-labels" + INFO
TARGETS = "targets" + INFO


def get_filename(pattern, name, args):
    filename = pattern.format(
        args.class_type, name, args.mode, args.threshold, args.sigma_gnmax,
        args.sigma_threshold, args.budget, args.transfer_type)
    return filename


def get_raw_queries_filename(name, args):
    return get_filename(pattern=RAW_QUERIES, name=name, args=args)


def get_queries_filename(name, args):
    return get_filename(pattern=QUERIES, name=name, args=args)


def get_aggregated_labels_filename(name, args):
    return get_filename(pattern=AGGREGATED_LABELS, name=name, args=args)


def get_targets_filename(name, args):
    return get_filename(pattern=TARGETS, name=name, args=args)


class QuerySet(Dataset):
    """Labeled dataset consisting of query-answer pairs."""

    def __init__(self, args, transform, id, target_transform=None):
        super(QuerySet, self).__init__()
        # Queries (the data points that was labeled).
        model_name = get_model_name_by_id(id=id)
        self.query_set_type = args.query_set_type
        if self.query_set_type == 'raw':
            filename = get_raw_queries_filename(name=model_name, args=args)
        elif self.query_set_type == 'numpy':
            filename = get_queries_filename(name=model_name, args=args)
        else:
            raise Exception(
                f"Unknown query set type for retraining: {self.query_set_type}")

        filepath = os.path.join(args.ensemble_model_path, filename)
        if os.path.isfile(filepath):
            self.samples = np.load(filepath)
        else:
            raise Exception(
                "Queries '{}' do not exist, please generate them via 'query_ensemble_model(args)'!".format(
                    filepath))
        # Answers to the queries (the labels assigned to the data points).
        filename = get_aggregated_labels_filename(name=model_name, args=args)
        filepath = os.path.join(args.ensemble_model_path, filename)
        if os.path.isfile(filepath):
            self.labels = np.load(filepath)
        else:
            raise Exception(
                "Answers '{}' do not exist, please generate them via 'query_ensemble_model(args)'!".format(
                    filepath))
        self.transform = transform
        self.target_transform = target_transform

        print('In QuerySet:')
        print('number of new labeled data points: ', len(self.labels))
        all_answers = self.labels.size
        print('number of all new items: ', all_answers)
        not_answered = np.sum(self.labels == np.nan)
        print('number of not answered: ', not_answered)
        print('number of answered: ', all_answers - not_answered)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.samples[idx]
        label = self.labels[idx]

        if self.query_set_type == 'raw':
            img = Image.fromarray(img)

        img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        else:
            label = torch.tensor(label)

        return img, label
