from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from utils.cifar_sampling import cifar_noniid
from utils.mnist_sampling import iid, mnist_noniid, mnist_noniid_dirichlet
from utils.tsrd_sampling import tsrd_noniid_dirichlet
from utils.init_models import nu_classes
from utils.imagenet_sampling import ImageFolder_custom
from PIL import Image
from collections import defaultdict
from utils.language_utils import word_to_indices, letter_to_vec
import torch
import os
import dill
import csv
import copy
import pickle
import json


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


class HARDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        if train==True:
            with open(data_path+"/train_x.pkl","rb") as f:
                self.data=dill.load(f)
            with open(data_path+"/train_y.pkl","rb") as f:
                self.targets=dill.load(f)
        else:
            with open(data_path+"/test_x.pkl","rb") as f:
                self.data=dill.load(f)
            with open(data_path+"/test_y.pkl","rb") as f:
                self.targets=dill.load(f)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
        


class TSRDDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(label_file, 'r') as f:
            lines = f.readlines()

        self.img_labels = [line.strip().split(';')[0] for line in lines]
        self.labels = []
        for line in lines:
            parts = line.strip().split(';')
            if parts[-2]:  # 检查字段是否为空
                self.labels.append(int(parts[-2]))
            else:
                # 这里可以处理空字段的情况，例如跳过它、记录错误或设置默认值
                print(f"Warning: Found an empty label in line: {line}")  # 打印警告信息


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
        

class SSpeare(Dataset):
    def __init__(self, train=True):
        super(SSpeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data("../data/shakespeare/train",
                                                                                 "../data/shakespeare/test")
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")





def data_assignment(args):
    print("!!!!!!!")
    freq_user = 1 / args.num_users

    n_classes = nu_classes(args)

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            #dict_users = iid(dataset_train, args.num_users)
            with open("./data/data_saved/mnist_iid_dataset.pkl","rb") as f:
                dict_users=dill.load(f)
        else:
            print("!!!!MNIST-noniid!!!!")
            dict_users = mnist_noniid_dirichlet(dataset=dataset_train, n_classes=10, num_users=args.num_users, beta=args.beta)
            #dict_users = mnist_noniid(dataset_train, args.num_users)
            #with open("./data/data_saved/mnist_noniid_dataset.pkl","rb") as f:
            #    dict_users=dill.load(f)
            #with open("./data/data_saved/mnist_noniid_dataset_old.pkl","rb") as f:
            #    dict_users=dill.load(f)

        #if args.iid:
        #    dict_users = iid(dataset_train, args.num_users)
        #else:
        #    dict_users = mnist_noniid(dataset_train, args.num_users)
            
        return dataset_train, dataset_test, dict_users, freq_user

    elif args.dataset == 'mnist2device':
        print("!!!!!!!")
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        with open("./data/data_saved/mnist_noniid_dataset_2device.pkl","rb") as f:
            dict_users=dill.load(f)
        print("!!!!!!!")
        return dataset_train, dataset_test, dict_users, freq_user

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.RandomCrop(32, 4),transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        
        # print(dataset_test.data.shape)
        
        #if args.iid:
        #    dict_users = iid(dataset_train, args.num_users)
        #else:
        #    dict_users, freq_user = cifar_noniid(dataset=dataset_train, n_classes=n_classes, num_users=args.num_users,
        #                                         main_class_number=args.main_class_number,
        #                                         beta=args.beta, imb=args.imb)
        
        #    with open("./data/data_saved/cifar_noniid.pkl","wb") as f:
        #        pickle.dump(dict_users, f)
            
        #    with open("./data/data_saved/cifar_noniid_freq.pkl","wb") as f:
        #        pickle.dump(freq_user, f)
        
        if args.iid:
        #    dict_users = cifar_iid(dataset_train, args.num_users)
            with open("./data/data_saved/cifar_iid_dataset.pkl","rb") as f:
                dict_users=dill.load(f)
            print('cifarIID')
        else:

#            dict_users, freq_user = cifar_noniid(dataset=dataset_train, n_classes=n_classes, num_users=args.num_users,
#                                                 main_class_number=args.main_class_number,
#                                                 beta=0.1, imb=args.imb)
            print('cifarnonIID')
#            with open("./data/data_saved/cifar_verynoniid.pkl","wb") as f:
#                pickle.dump(dict_users, f)
#            with open("./data/data_saved/cifar_verynoniid_freq.pkl","wb") as f:
#                pickle.dump(freq_user, f)

#            with open("./data/data_saved/cifar_verynoniid.pkl","rb") as f:
#                dict_users=dill.load(f)
#        
#            with open("./data/data_saved/cifar_verynoniid_freq.pkl","rb") as f:
#                freq_user=dill.load(f)
        
            with open("./data/data_saved/cifar_noniid.pkl","rb") as f:
                dict_users=dill.load(f)
        
            with open("./data/data_saved/cifar_noniid_freq.pkl","rb") as f:
                freq_user=dill.load(f)
                
        #    freq_user = {fre_i: [] for fre_i in range(args.num_users)}
        #    with open('./data/data_saved/cifar_noniid_dataset_freq.csv', 'r') as f:
        #        csvread = csv.reader(f)
        #        idx = 0
        #        for row in csvread:
        #            for column in range(10):
        #                freq_user[idx].append(int(row[column]))
        #            idx += 1

        #    for fid in freq_user:
        #        cover = dict(zip(range(10), copy.deepcopy(freq_user[fid])))
        #        freq_user[fid] = cover
        #    print(freq_user)
                                        
                                                 
        return dataset_train, dataset_test, dict_users, freq_user
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose([transforms.RandomCrop(32, 4),transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
        # print(dataset_test.data.shape)
        if args.iid:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users, freq_user = cifar_noniid(dataset=dataset_train, n_classes=n_classes, num_users=args.num_users,
                                                 main_class_number=args.main_class_number,
                                                 beta=args.beta, imb=args.imb)
        return dataset_train, dataset_test, dict_users, freq_user

    elif args.dataset == 'tinyimagenet':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = ImageFolder_custom(root='../data/tiny-imagenet-200/train/', transform=transform)
        dataset_test = ImageFolder_custom(root='../data/tiny-imagenet-200/val/', transform=transform)
        if args.iid:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users, freq_user = cifar_noniid(dataset=dataset_train, n_classes=n_classes, num_users=args.num_users,
                                                 main_class_number=args.main_class_number,
                                                 beta=args.beta, imb=args.imb)
        return dataset_train, dataset_test, dict_users, freq_user
        
        
    elif args.dataset == 'har':
        dataset_train=HARDataset('../data/har',train=True,transform=None)
        dataset_test=HARDataset('../data/har',train=False,transform=None)
        with open("./data/data_saved/har_dataset.pkl","rb") as f:
            dict_users=dill.load(f)
        return dataset_train, dataset_test, dict_users, freq_user
            
    elif args.dataset == 'tsrd':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # You might want to adjust this depending on TSRD image sizes.
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        dataset_train = TSRDDataset(img_dir='../data/TSRD/tsrd-train', label_file='../data/TSRD/TsignRecgTrain4170Annotation.txt', transform=transform)
        dataset_test = TSRDDataset(img_dir='../data/TSRD/TSRD-Test', label_file='../data/TSRD/TsignRecgTest1994Annotation.txt', transform=transform)
        #dict_users, freq_user = tsrd_noniid_dirichlet(dataset=dataset_train, n_classes=58, num_users=args.num_users, beta=args.beta)
        #with open("./data/data_saved/tsrd_noniid.pkl","wb") as f:
        #    pickle.dump(dict_users, f)
            
        #with open("./data/data_saved/tsrd_noniid_freq.pkl","wb") as f:
        #    pickle.dump(freq_user, f)
        
        with open("./data/data_saved/tsrd_noniid.pkl","rb") as f:
            dict_users=dill.load(f)
        
        with open("./data/data_saved/tsrd_noniid_freq.pkl","rb") as f:
            freq_user=dill.load(f)

        return dataset_train, dataset_test, dict_users, freq_user
        
    elif args.dataset == 'shakespeare':
        dataset_train = SSpeare(train=True)
        dataset_test = SSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        return dataset_train, dataset_test, dict_users, freq_user

    else:
        exit('Error: unrecognized dataset' + str(args.dataset))






