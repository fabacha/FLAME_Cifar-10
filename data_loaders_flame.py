from __future__ import print_function
import numpy as np
import random
import math
import os
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import torch.multiprocessing as mp

# Set the sharing strategy to 'file_system' to avoid "Too many open files" error
mp.set_sharing_strategy('file_system')

def get_shapes(dataset):
    if dataset == 'CIFAR-10':
        num_inputs = 3 * 32 * 32
        num_outputs = 10
        num_labels = 10
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels


def load_data(dataset, seed):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset == 'CIFAR-10':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2,
                                                   worker_init_fn=seed_worker)

        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2,
                                                  worker_init_fn=seed_worker)
   
    else:
        raise NotImplementedError
    return train_loader, test_loader


def assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="HAR", seed=1):
    other_group_size = (1 - bias) / (num_labels - 1)
    if dataset == "CIFAR-10":
        worker_per_group = num_workers / num_labels

    else:
        raise NotImplementedError

    # assign training data to each worker
    if dataset == "CIFAR-10":

        each_worker_data = [[] for _ in range(num_workers)]

        each_worker_label = [[] for _ in range(num_workers)]
    else:
        raise NotImplementedError
    server_data = []
    server_label = []

    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])
    server_counter = [0 for _ in range(num_labels)]

    # compute the labels need for each class

    if dataset == 'CIFAR-10':
        for batch_idx, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            for (x, y) in zip(data, label):
                # Assign x and y to the appropriate client or server based on the method in the original code
                if server_counter[int(y.cpu().numpy())] < samp_dis[int(y.cpu().numpy())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.cpu().numpy())] += 1
                else:
                    worker_index = random.sample(range(num_workers), 1)[0]
                    each_worker_data[worker_index].append(x)
                    each_worker_label[worker_index].append(y)

    else:
        raise NotImplementedError

    if server_pc != 0:
        server_data = torch.stack(server_data, dim=0)
        server_label = torch.stack(server_label, dim=0)
    else:
        if dataset == "CIFAR-10":
            server_data = torch.empty(size=(0, 3, 32, 32)).to(device)

        else:
            raise NotImplementedError

    server_label = torch.empty(size=(0,)).to(device)

    each_worker_data = [torch.stack(each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [torch.stack(each_worker, dim=0) for each_worker in each_worker_label]

    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return server_data, server_label, each_worker_data, each_worker_label
