#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import numpy as np
from torchvision import datasets, transforms

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # frequency per class
    freq_pool = {fre_i: None for fre_i in range(num_users)}

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])


    return dict_users



def cifar_noniid_imbalanced(dataset, n_classes, num_users, num_main_class):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_classes = 10
    labels = np.array(dataset.targets)
    num_training = labels.shape[0]

    idx_shard = [i for i in range(num_training)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_training)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    num_other_class = 5000 - num_main_class
    rand_class = 9

    # frequency per class
    freq_pool = {fre_i: [] for fre_i in range(num_users)}

    for i in range(num_users):
        class_pool = {c_i: 0 for c_i in range(num_classes)}
        class_pool[rand_class] = num_main_class
        freq_pool[i] = class_pool
        # choose main class
        rand_set = set(np.random.choice(idx_shard[5000*rand_class:5000*(rand_class +1)], num_main_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for j in rand_set:
            dict_users[i] = np.append(dict_users[i], idxs[j])
        rand_class -= 1

    for i in range(num_users):
        class_pool_r = freq_pool[i]
        # choose other class
        rand_set_other = set(np.random.choice(idx_shard, num_other_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set_other)
        for j in rand_set_other:
            dict_users[i] = np.append(dict_users[i], idxs[j])
            c_id = idxs[j]//5000
            class_pool_r[c_id] += 1

    for key in dict_users:
        vector = dict_users[key]
        vector = np.array(vector, dtype=int)
        dict_users[key] = vector

    Final_freq_pool = copy.deepcopy(freq_pool)
    for ukey, uvalue in freq_pool.items():
        cvector = []
        for c_idi in uvalue:
            cvector.append(uvalue[c_idi])
        Final_freq_pool[ukey] = cvector
    # print(Final_freq_pool)

    record_net_data_stats(labels, dict_users)

    return dict_users, Final_freq_pool
'''
    for i in range(num_users):
        class_pool = {c_i: None for c_i in range(num_classes)}
        rand_set = set(np.random.choice(idx_shard[5000*rand_class:5000*(rand_class +1)], num_main_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for j in rand_set:
            dict_users[i] = np.append(dict_users[i], idxs[j])
        rand_class -= 1
        class_pool
        # rand_set2 = set(
        #     np.random.choice(idx_shard[5000 * rand_class:5000 * (rand_class + 1)], num_main_class, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set2)
        # for j in rand_set2:
        #     dict_users[i] = np.append(dict_users[i], idxs[j])
        # rand_class -= 1
    for i in range(num_users):
        class_pool = {c_i: None for c_i in range(num_classes)}
        rand_set = set(np.random.choice(idx_shard, num_other_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for j in rand_set:
            dict_users[i] = np.append(dict_users[i], idxs[j])
    '''


def cifar_noniid_dirichlet(dataset, n_classes, num_users, beta=0.1):

    dict_users = {i: np.array([]) for i in range(num_users)}
    # frequency per class
    freq_pool = {fre_i: [] for fre_i in range(num_users)}
    idx_batch = [[] for _ in range(num_users)]


    labels = np.array(dataset.targets)
    num_sample_per_client = labels.shape[0] / num_users
    num_classes = 10

    min_size = 0
    min_require_size = 10


    for k in range(num_classes):

        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # the proportion of each client for class k
        proportions = np.random.dirichlet(np.repeat(beta, num_users))
        # balanced
        proportions = np.array([p * (len(idx_j) < num_sample_per_client) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        count = 0
        for c_id, idx_count in zip(range(num_users), proportions):
            freq_pool[c_id].append(copy.deepcopy(idx_count - count))
            count = copy.deepcopy(idx_count)
        freq_pool[num_users-1].append(5000 - proportions[-1])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    for fid in freq_pool:
        cover = dict(zip(range(num_classes), copy.deepcopy(freq_pool[fid])))
        freq_pool[fid] = cover
    print(freq_pool)

#     record_net_data_stats(labels, dict_users)
    return dict_users, freq_pool



def cifar_noniid(dataset, n_classes, num_users, main_class_number, beta=0.1, imb=False):
    if imb:
        dict_users, freq_user = cifar_noniid_imbalanced(dataset, n_classes, num_users, main_class_number)
    else:
        dict_users, freq_user = cifar_noniid_dirichlet(dataset, n_classes, num_users, beta)

    return dict_users, freq_user


def auxiliary_data(dataset, num_main):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_classes = 10
    labels = np.array(dataset.targets)
    num_training = labels.shape[0]

    idx_shard = [i for i in range(num_training)]

    dict_users = {i: np.array([]) for i in range(num_classes)}
    idxs = np.arange(num_training)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    rand_class = 9

    # assignment for main class
    for i in range(num_classes):

        rand_set = set(np.random.choice(idx_shard[100 * rand_class:100 * (rand_class + 1)], 32, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for j in rand_set:
            dict_users[i] = np.append(dict_users[i], idxs[j])
        rand_class -= 1

    temp_vector = None
    for key in dict_users:
        vector = dict_users[key]
        vector = np.array(vector, dtype=int)
        if key == 0:
            temp_vector = vector
        else:
            temp_vector = np.concatenate((temp_vector, vector), axis=None)

    return temp_vector



if __name__ == '__main__':
    trans_cifar = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)

    num = 10
    # d = cifar_noniid_dirichlet(dataset_train, num)

    num_main = 32
    vector = auxiliary_data(dataset_test, num_main)
    print(vector)
    with open('100_auxiliary.npy', 'wb') as f:
        np.save(f, vector)
    with open('100_auxiliary.npy', 'rb') as f:
        a = np.load(f)