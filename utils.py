import os
import numpy as np
import pandas as pd
import netCDF4 as nc

import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
import einops

def ramp_up(min_v, max_v, cur_t, MAX_T):
    cur_t = min(cur_t, MAX_T)
    return (max_v - min_v) / MAX_T * cur_t + min_v


def mse_loss_with_nan(x, y, mask):
    y = torch.nan_to_num(y)
    loss = F.mse_loss(x, y, reduction='none')
    loss = (loss * mask).sum() / (mask.sum() + 1e-3)
    return loss


def likelihood_with_mask(d, y, mask):
    y = torch.nan_to_num(y)
    p = -d.log_prob(y)
    p = (p * mask).sum() / (mask.sum() + 1e-3)
    return p


def kl_div_with_mask(p, q, mask):
    kl_div = kl_divergence(p, q)
    kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-3)
    return kl_div


def augment(x, method='mask', intensity=0.1):
    def _mask(x):
        mask = torch.rand_like(x)
        mask = (mask < 1 - intensity).float()
        return x * mask

    def _shuffle(x):
        index = torch.randperm(x.size(-1)).to(x.device)
        perm_x = torch.index_select(x, -1, index)

        return x * (1 - intensity) + perm_x * intensity

    cat = x[..., :4]
    num = x[..., 4:]

    if method == 'mask':
        num = _mask(num)
    if method == 'shuffle':
        num = _shuffle(num)

    x = torch.cat([cat, num], dim=-1)
    return x

