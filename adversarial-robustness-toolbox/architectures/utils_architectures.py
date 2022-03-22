import pickle

import os
import torch
from torch import nn as nn

from models.private_model import get_private_model


def pytorch2pickle(args):
    model_types = args.architectures
    nr_model_types = len(model_types)
    for id in range(args.num_models):
        model_type = model_types[id % nr_model_types]
        args.architecture = model_type
        args.private_model_path = os.path.join(
            args.path, 'private-models',
            args.dataset, model_type, '{:d}-models'.format(
                args.num_models))
        filename = "checkpoint-{}.pth.tar".format('model({:d})'.format(id + 1))
        filepath = os.path.join(args.private_model_path, filename)
        if os.path.isfile(filepath):
            # Load private model
            checkpoint = torch.load(filepath)
            model = get_private_model(name='model({:d})'.format(id + 1), args=args,
                                      model_type=model_type)
            model.load_state_dict(checkpoint['state_dict'])
            # Copy weights
            conv_count = 0
            bn_count = 0
            linear_count = 0
            weights = {}
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    key_weight = 'conv_{}_weight'.format(conv_count)
                    weights[key_weight] = m.weight.data.clone().cpu().numpy()
                    conv_count += 1
                elif isinstance(m, nn.BatchNorm2d):
                    key_gamma = 'bn_{}_gamma'.format(bn_count)
                    key_beta = 'bn_{}_beta'.format(bn_count)
                    key_rm = 'bn_{}_rm'.format(bn_count)
                    key_rv = 'bn_{}_rv'.format(bn_count)
                    weights[key_gamma] = m.weight.data.clone().cpu().numpy()
                    weights[key_beta] = m.bias.data.clone().cpu().numpy()
                    weights[key_rm] = m.running_mean.data.clone().cpu().numpy()
                    weights[key_rv] = m.running_var.data.clone().cpu().numpy()
                    bn_count += 1
                elif isinstance(m, nn.Linear):
                    key_weight = 'linear_{}_weight'.format(linear_count)
                    key_bias = 'linear_{}_bias'.format(linear_count)
                    weights[key_weight] = m.weight.data.clone().cpu().numpy()
                    weights[key_bias] = m.bias.data.clone().cpu().numpy()
                    linear_count += 1
            # Serialize weight dictionary
            filename = "checkpoint-{}.pickle".format(model.name)
            filepath = os.path.join(args.private_model_path, filename)
            with open(filepath, 'wb') as handle:
                pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Model {}'s weights serialized with success!".format(
                model.name))
        else:
            raise Exception(
                "Checkpoint file '{}' does not exist, please generate it via 'train_private_models(args)'!".format(
                    filepath))
