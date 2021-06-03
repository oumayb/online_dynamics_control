import torch.nn as nn

from odc.models import CNNAE


def get_loss(name, config=None):
    if name == "MSE":
        return mse_loss(reduction=config['loss_reduction'])


def get_autoencoder(name, config=None):
    if name == "CNNAE":
        autoenc = CNNAE(config=config)
    return autoenc


def mse_loss(reduction):
    def mse_loss_(output, x):
        loss = nn.MSELoss(reduction=reduction)
        return loss(x, output['recon'])
    return mse_loss_


def _mse_loss(output, x):
    loss = nn.MSELoss()
    return loss(x, output['recon'])


def make_gradients_dict(model):
    """
    Params
    ------
    model: nn.module

    Returns
    -------
    gradients: dict:
        - keys: module names
        - values: dict:
            - keys: "min", "max", "avg"
    """
    gradients = dict()

    for name, _ in model.named_modules():
        if (name != "") and (name !="autoencoder") and (name !='relu') and (name!='autoencoder.relu'):
            gradients[name] = dict()

    for name in gradients.keys():
        gradients[name]['max'] = []
        gradients[name]['min'] = []

    return gradients


def my_register_backward_hook(model, module, name):
    hook = make_hook_function(name=name, model=model)
    module.register_backward_hook(hook)


def make_hook_function(name, model):
    def save_gradients(module, grad_input, grad_output):
        model.gradients[name]['max'].append(grad_output[0].max())
        model.gradients[name]['min'].append(grad_output[0].min())
    return save_gradients
