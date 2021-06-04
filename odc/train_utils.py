import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def get_loss(name, config=None):
    if name == "MSE":
        return mse_loss(reduction=config['loss_reduction'])


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


def plot_for(all_values, n_cols=5, plot_func= plt.plot, figsize=(10, 10), titles=None, **kwargs):
    fig = plt.figure(figsize=figsize);
    n_rows = math.ceil(len(all_values) / n_cols)
    row = 1
    counter = 0
    col = 1
    for i, values in enumerate(all_values):
        f = plt.subplot(n_rows, n_cols, i + 1);
        f.set_axis_off()
        plot_func(values, **kwargs);
    if titles is not None:
        plt.title(titles[i]);
    counter += 1
    col += 1
    if counter % n_cols == 0:
        row += 1
        col = 1


def save_gif(frames, fps, target_path):
    try:
        h, w = frames[0].shape
        n_c = 0
    except:
        h, w, n_c = frames[0].shape

    if n_c == 3:
        clip = ImageSequenceClip(sequence=list(frames.numpy()), fps=fps)
        clip.write_gif(target_path)
    else:
        frames_RGB = [np.stack((pic, pic, pic), axis=2) for pic in frames]
        clip = ImageSequenceClip(sequence=frames_RGB, fps=fps)
        clip.write_gif(target_path)