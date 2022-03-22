import torch
import os
from models.private_model import get_private_model_by_id
from models.utils_models import get_model_name_by_id


def load_private_models(args, model_path=None):
    # Load private models
    private_models = []

    for id in range(args.num_models):
        private_model = load_private_model_by_id(
            args=args, id=id, model_path=model_path)
        private_models.append(private_model)

    return private_models


def load_victim_model(args):
    private_model = load_private_model_by_id(
        args=args, id=0, model_path=args.victim_model_path)
    return private_model


def load_private_model_by_id(args, id, model_path=None):
    """
    Load a single model by its id.
    :param args: program parameters
    :param id: id of the model
    :return: the instance of the model
    """
    model_name = get_model_name_by_id(id=id)
    if args.dataset == 'pascal':
        filename = f"multilabel_net_params_{id}.pkl"
    else:
        filename = f"checkpoint-{model_name}.pth.tar"
    if model_path is None:
        model_path = args.save_model_path

    filepath = os.path.join(model_path, filename)

    if os.path.isfile(filepath):
        model = get_private_model_by_id(args=args, id=id)
        checkpoint = torch.load(filepath)
        if args.dataset == 'pascal':
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        if args.cuda:
            model.cuda()
        if 'taus' in checkpoint and args.load_taus == 'apply':
            taus = checkpoint['taus']
            # The output from the model will be the normalized sigmoid.
            model.set_op_threshs(op_threshs=taus)
            args.sigmoid_op = 'disable'
            args.multilabel_prob_threshold = [0.5]
        if 'label_weights' in checkpoint and args.label_reweight is 'apply':
            model.label_weights = checkpoint['label_weights']
        model.eval()
        return model
    else:
        raise Exception(
            f"Checkpoint file {filepath} does not exist, please generate it via "
            f"train_private_models(args)!")
