import copy

import numpy as np
from torchvision import datasets, transforms

def tsrd_noniid_dirichlet(dataset, n_classes, num_users, beta=0.5):

    dict_users = {i: np.array([]) for i in range(100)}
    freq_pool = {fre_i: [] for fre_i in range(100)}
    idx_batch = [[] for _ in range(50)]

    # Here we're extracting labels from TSRD dataset.
    # This assumes your TSRDDataset has a 'labels' attribute after initializing.
    labels = np.array(dataset.labels)
    num_sample_per_client = len(labels) / 50
    num_classes = n_classes

    min_size = 0
    min_require_size = 10

    for k in range(num_classes):

        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(beta, 50))
        proportions = np.array([p * (len(idx_j) < num_sample_per_client) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        count = 0
        for c_id, idx_count in zip(range(50), proportions):
            freq_pool[c_id].append(copy.deepcopy(idx_count - count))
            count = copy.deepcopy(idx_count)
        freq_pool[49].append(len(idx_k) - proportions[-1])  # Adjust this value based on your dataset size.

    for j in range(50):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        
    for j in range(50, 100):
        np.random.shuffle(idx_batch[j - 50])
        dict_users[j] = idx_batch[j - 50]

    for fid in freq_pool:
        if fid < 50:
            cover = dict(zip(range(num_classes), copy.deepcopy(freq_pool[fid])))
            freq_pool[fid] = cover
        else:
            freq_pool[fid] = freq_pool[fid - 50]
    print(freq_pool)

    return dict_users, freq_pool
