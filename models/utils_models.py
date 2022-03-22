from enum import Enum


class model_size(Enum):
    small = 'small'
    big = 'big'

    def __str__(self):
        return self.name


def set_model_size(args):
    # Set model_size only if it is None.
    if args.model_size is not None:
        return

    small_models = {'VGG3', 'VGG5', 'VGG7', 'VGG9',
                    'ResNet6', 'ResNet8', 'ResNet10', 'ResNet12',
                    'MnistNet', 'MnistNetPate', 'FashionMnistNet'
                    }

    for model_type in args.architectures:
        if model_type not in small_models:
            # if one of the models is big then we the model size in general is big
            args.model_size = model_size.big
            return

    args.model_size = model_size.small
    return


def get_model_type_by_id(args, id):
    model_types = args.architectures
    nr_model_types = len(model_types)
    model_type = model_types[id % nr_model_types]
    return model_type


def get_model_name_by_id(id):
    name = 'model({:d})'.format(id + 1)
    return name


