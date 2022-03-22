def get_config_optim(model, lr, weight_decay):
    params_dict = dict(model.named_parameters())
    params = []

    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key or 'adj' == key else 1.0

        if key.startswith('conv1') or key.startswith('bn1'):
            lr_mult = 0.1
        elif 'fc' in key:
            lr_mult = 1.0
        elif 'adj' == key:
            lr_mult = 0.0
        elif 'gc' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.1

        params.append({'params': value,
                       'lr': lr,
                       'lr_mult': lr_mult,
                       'weight_decay': weight_decay,
                       'decay_mult': decay_mult})

    return params
