from architectures.fashion_mnist import FashionMnistNet
from architectures.mnist_net import MnistNet
from architectures.mnist_net_pate import MnistNetPate
from architectures.resnet import ResNet10, ResNet12, ResNet14, ResNet16, \
    ResNet18, ResNet34
from architectures.small_resnet import ResNet8
from architectures.tiny_resnet import ResNet6
from architectures.vggs import VGG
from models.utils_models import get_model_type_by_id, get_model_name_by_id

def get_private_model(name, model_type, args):
    """Private model held by each party."""
    if model_type.startswith('VGG'):
        model = VGG(name=name, args=args, model_type=model_type)
    elif model_type == 'ResNet6':
        model = ResNet6(name=name, args=args)
    elif model_type == 'ResNet8':
        model = ResNet8(name=name, args=args)
    elif model_type == 'ResNet10':
        model = ResNet10(name=name, args=args)
    elif model_type == 'ResNet12':
        model = ResNet12(name=name, args=args)
    elif model_type == 'ResNet14':
        model = ResNet14(name=name, args=args)
    elif model_type == 'ResNet16':
        model = ResNet16(name=name, args=args)
    elif model_type == 'ResNet18':
        model = ResNet18(name=name, args=args)
    elif model_type in ['ResNet34', 'ResNet34-ood']:
        model = ResNet34(name=name, args=args)
    elif model_type == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=20) 
    elif model_type == 'MnistNet':
        model = MnistNet(name=name, args=args)
    elif model_type == 'MnistNetPate':
        model = MnistNetPate(name=name, args=args)
    elif model_type == 'FashionMnistNet':
        model = FashionMnistNet(name=name, args=args)
    else:
        raise Exception(f'Unknown architecture: {model_type}')

    # Set the attributes if not already set.
    if getattr(model, 'dataset', None) == None:
        model.dataset = args.dataset
    if getattr(model, 'model_type', None) == None:
        model.model_type = model_type

    return model


def get_private_model_by_id(args, id=0):
    model_type = get_model_type_by_id(args=args, id=id)
    name = get_model_name_by_id(id=id)
    model = get_private_model(name=name, args=args, model_type=model_type)
    return model
