#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=1500, help="rounds of communication")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_iter', type=int, default=10, help="the number of local iterations: H")
    parser.add_argument('--unit', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lr_sche', type=str, default='fix', help="LR schedule")
    parser.add_argument('--optim', type=str, default='sgd', help="LR schedule")

    # contrastive arguments
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--mu', type=float, default=5, help='the mu parameter for fedprox or moon')
    parser.add_argument('--t', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--dim', default=400, type=int, help="dimension of subvector sent to infoNCE")
    parser.add_argument('--mloss', type=str, default="moon", help='choice of loss')  # "avg", "moon", "DGA", "PIPE"
    parser.add_argument('--alpha', type=float, default=0.1, help='momentum_based updates')

    # delay parameters
    parser.add_argument('--delay', type=int, default=1, help='the number to receive delay global models')
    parser.add_argument('--glob_check', type=int, default=5, help='check point for global models')

    # models arguments
    parser.add_argument('--models', type=str, default="resnet", help='models name')  # model_con

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='tinyimagenet', help="name of dataset")  # cifar
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")

    parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')  # action='store_true',
    parser.add_argument('--imb', type=bool, default=False, help="data partition: class imbalanced or dirichlet distribution")
    parser.add_argument('--beta', type=float, default=0.5, help="parameter of dirichlet distribution")
    parser.add_argument('--main_class_number', type=int, default=3500, help="number of main class samples")
    parser.add_argument('--m_class_number', type=int, default=2, help="number of main class samples")

    # other arguments
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--T', type=float, default=30)
    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--b', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--fusion_weight', type=float, default=0.8, help='fusion weight of global model in new overlap round')
    parser.add_argument('--tao', type=float, default=0.1, help='increasing scale of local iter')
    parser.add_argument('--setK',type=int, default=0)
    
    parser.add_argument('--id', type=str)
    args = parser.parse_args()

    return args
