import copy
import numpy as np
from torchvision import datasets, transforms
import pickle
from torchvision.datasets import ImageFolder, DatasetFolder


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


def imagenet_noniid_dirichlet(dataset, n_classes, num_users, beta=0.1):
    X_train, labels = np.array([s[0] for s in dataset.samples]), \
                       np.array( [int(s[1]) for s in dataset.samples])

    dict_users = {i: np.array([]) for i in range(num_users)}
    # frequency per class
    freq_pool = {fre_i: [] for fre_i in range(num_users)}
    idx_batch = [[] for _ in range(num_users)]


    # labels = np.array(dataset.targets)
    num_sample_per_client = labels.shape[0] / num_users
    num_classes = n_classes

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
        freq_pool[num_users-1].append(500 - proportions[-1])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    for fid in freq_pool:
        cover = dict(zip(range(num_classes), copy.deepcopy(freq_pool[fid])))
        freq_pool[fid] = cover
    print(freq_pool)

#     record_net_data_stats(labels, dict_users)
    return dict_users, freq_pool


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dl_obj = ImageFolder_custom
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_train = dl_obj(root='/home/c3_server2/model_contrastive/data/tiny-imagenet-200/train/',
                                       transform=transform)
    dataset_test = dl_obj(root='/home/c3_server2/model_contrastive/data/tiny-imagenet-200/val/',
                                      transform=transform)

    num = 200
    num_users = 128
    dict_users, freq_user = imagenet_noniid_dirichlet(dataset=dataset_train, n_classes=num,
                                                      num_users=num_users)

    with open('128_user_dict.pkl', 'wb') as f:
        pickle.dump(dict_users, f)

    with open('128_user_freq.pkl', 'rb') as f:
        pickle.dump(freq_user, f)