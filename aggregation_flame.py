import attacks_flame


import math
import os
import collections
from functools import reduce

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import hdbscan
import copy
import utils

# Copyright (c) 2015, Leland McInnes
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Available at: https://github.com/scikit-learn-contrib/hdbscan

def fedavg(gradients, net, lr, f, byz, device, data_sizes):
    """
    Based on the description in https://arxiv.org/abs/1602.05629
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    data_size: amount of training data of each worker device
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks_flame.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)

    n = len(param_list)
    total_data_size = sum(data_sizes)

    # compute global model update
    global_update = torch.zeros(param_list[0].size()).to(device)
    for i, grad in enumerate(param_list):
        global_update += grad * data_sizes[i]
    global_update /= total_data_size

    # update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)


def flame(gradients, net, lr, f, byz, device, epsilon, delta):
    """
    Based on the description in https://arxiv.org/abs/2101.02281
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    epsilon: parameter for differential privacy
    delta: parameter for differential privacy
    """
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    if byz == attacks_flame.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    n = len(param_list)

    # compute pairwise cosine distances
    cos_dist = torch.zeros((n, n), dtype=torch.double).to(device)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - F.cosine_similarity(param_list[i], param_list[j], dim=0, eps=1e-9)
            cos_dist[i, j], cos_dist[j, i] = d, d

    # clustering of gradients
    np_cos_dist = cos_dist.cpu().numpy()
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=1, min_cluster_size=(n // 2) + 1,
                                cluster_selection_epsilon=0.0, allow_single_cluster=True).fit(np_cos_dist)

    # compute clipping bound
    euclid_dist = []
    for grad in param_list:
        euclid_dist.append(torch.norm(lr * grad, p=2))

    clipping_bound, _ = torch.median(torch.stack(euclid_dist).reshape((-1, 1)), dim=0)

    # gradient clipping
    clipped_gradients = []
    for i in range(n):
        if clusterer.labels_[i] == 0:
            gamma = clipping_bound / euclid_dist[i]
            clipped_gradients.append(-lr * param_list[i] * torch.min(torch.ones((1,)).to(device), gamma))

    # aggregation
    global_update = torch.mean(torch.cat(clipped_gradients, dim=1), dim=-1)

    # adaptive noise
    std = (clipping_bound * np.sqrt(2 * np.log(1.25 / delta)) / epsilon) ** 2
    global_update += torch.normal(mean=0, std=std.item(), size=tuple(global_update.size())).to(device)

    # update the global model
    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())))
        idx += torch.numel(param)
