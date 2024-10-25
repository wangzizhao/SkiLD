import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_layer(w, b):
    fan_in = w.shape[-2]
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(w, -bound, bound)
    nn.init.uniform_(b, -bound, bound)


def reset_layer_orth(w, b):
    nn.init.orthogonal_(w)
    nn.init.zeros_(b)


def forward_network(x, weights, biases=None, activation=F.relu):
    """
    given an input and a multi-layer networks (i.e., a list of weights and a list of biases),
        apply the network to each input, and return output
    the same activation function is applied to all layers except for the last layer
    """
    x_ndim = x.ndim
    p_bs = x.shape[:-2]
    reshape = len(p_bs) > 1
    if reshape:
        x = x.reshape(-1, *x.shape[-2:])

    if isinstance(weights, nn.Parameter):
        weights = [weights]
        if biases is not None:
            biases = [biases]

    if biases is None:
        biases = [None] * len(weights)

    for i, (w, b) in enumerate(zip(weights, biases)):
        # x (p_bs, bs, in_dim), bs: data batch size which must be 1D
        # w (p_bs, in_dim, out_dim), p_bs: parameter batch size
        # b (p_bs, 1, out_dim)
        assert x_ndim == w.ndim

        if p_bs != w.shape[:-2]:
            w = w.expand(*p_bs, -1, -1)
            if b is not None:
                b = b.expand(*p_bs, -1, -1)

        if reshape:
            w = w.reshape(-1, *w.shape[-2:])
            if b is not None:
                b = b.reshape(-1, *b.shape[-2:])

        x = torch.bmm(x, w)
        if b is not None:
            x = x + b

        if i < len(weights) - 1 and activation:
            x = activation(x)

    if reshape:
        x = x.reshape(*p_bs, *x.shape[-2:])

    return x


def expand_helper(x, dim, size):
    if x is None:
        return x
    elif isinstance(x, torch.Tensor):
        x = x.unsqueeze(dim)
        expand_size = [-1] * x.ndim
        expand_size[dim] = size
        return x.expand(expand_size)
    elif isinstance(x, (list, tuple)):
        return [expand_helper(x_i, dim, size) for x_i in x]
    elif isinstance(x, dict):
        return {k: expand_helper(v, dim, size) for k, v in x.items()}
    else:
        raise NotImplementedError("Unknown input type {}".format(type(x)))


def flatten_helper(x, dim):
    if x is None:
        return x
    elif isinstance(x, torch.Tensor):
        return x.reshape(-1, *x.shape[dim:])
    elif isinstance(x, (list, tuple)):
        return [flatten_helper(x_i, dim) for x_i in x]
    else:
        raise NotImplementedError("Unknown input type {}".format(type(x)))


def mixup_helper(x, mix_weight, rand_idx):
    if x is None:
        return x
    elif isinstance(x, torch.Tensor):
        return x * mix_weight + x[rand_idx] * (1 - mix_weight)
    elif isinstance(x, (list, tuple)):
        return [mixup_helper(x_i, mix_weight, rand_idx) for x_i in x]
    else:
        raise NotImplementedError("Unknown input type {}".format(type(x)))
