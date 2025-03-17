from models.encoder_head import *
import numpy as np

def nu_classes(args):

    n_classes = 0
    if args.dataset in {'mnist', 'cifar', 'svhn', 'fmnist','mnist2device'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    elif args.dataset == 'har':
        n_classes = 6
    elif args.dataset == 'tsrd':
        n_classes = 58
    elif args.dataset == 'shakespeare':
        n_classes = 80
    else:
        exit('Error: unrecognized dataset')
    return n_classes

def init_nets(n_parties, args, get_initial=False):
    # n_parties = args.num_users
    net_configs = args.net_config

    nets = {net_i: None for net_i in range(n_parties)}
    n_classes = nu_classes(args)
    dim = args.out_dim

    if args.dataset == 'cifar' and args.models == 'resnet':
        model_name = "resnet18-cifar10"
    elif args.dataset == 'cifar' and args.models == 'cnn':
        model_name = "cnn-cifar"
    elif args.dataset == 'mnist' and args.models == 'cnn':
        model_name = "cnn-mnist"
        dim = 30
    elif args.dataset == 'har' and args.models == 'cnn':
        model_name = "cnn-har"
        dim = 64
    elif args.dataset == 'cifar' and args.models == 'squeezenet':
        model_name = 'squeezenet-cifar'
        dim = args.out_dim
    elif args.dataset == 'tsrd' and args.models == 'mobilenet':
        model_name = 'mobilenet-tsrd'
    elif args.dataset == 'shakespeare' and args.models == 'lstm':
        model_name = 'lstm-shakespeare'
        dim = args.out_dim    
    elif args.dataset == 'cifar' and args.models == 'mobile':
        model_name = "mobilenet"
    elif args.dataset == 'tsrd' and args.models == 'mobileformer':
        model_name = "mobileformer-tsrd"
        dim = 1280
    elif args.dataset == 'tinyimagenet' and args.models == 'resnet':
        model_name = "resnet50-imagenet"
    elif args.dataset == 'mnist2device' and args.models == 'cnn':
        model_name = "cnn-mnist"
        dim = 30
    else:
        exit('Error: unrecognized models and datasets (init_models.py)')


    for net_i in range(n_parties):
        if 'moon' in args.mloss and args.models == 'cnn':
            net = ModelFedCon_cnn(base_model=model_name, out_dim=dim, n_classes=n_classes).to(args.device)
        elif 'moon' in args.mloss and args.models == 'squeezenet':
            net = ModelFedCon_squeezenet(base_model=model_name, out_dim=dim, n_classes=n_classes).to(args.device)
        elif 'moon' in args.mloss and args.models == 'mobilenet':
            net = ModelFedCon_mobilenet(base_model=model_name, out_dim=dim, n_classes=n_classes).to(args.device)
        elif 'moon' in args.mloss and args.models == 'lstm':
            net = ModelFedCon_lstm(base_model=model_name, out_dim=dim, n_classes=n_classes).to(args.device)
        elif 'moon' in args.mloss and args.models == 'mobileformer':
            net = ModelFedCon_mobileformer(base_model=model_name, out_dim=dim, n_classes=n_classes).to(args.device)          
        elif ('prox' in args.mloss or 'CR' in args.mloss) and args.models != 'cnn':
            net = ModelFedCon_noheader(base_model=model_name, n_classes=n_classes).to(args.device)
        elif ('prox' in args.mloss or 'CR' in args.mloss) and args.models == 'cnn':
            net = ModelFedCon_cnn_noheader(base_model=model_name, n_classes=n_classes).to(args.device)
        nets[net_i] = net


    X = np.zeros((args.epochs, 6))
    X_align = np.zeros((args.epochs, args.num_users))
    X_cka = np.zeros((args.num_users, args.num_users + 1))
    X_neg = np.zeros((args.epochs, args.num_users))
    X_neg1 = np.zeros((args.epochs, args.num_users))

    if get_initial:
        return nets, X, X_align, X_neg, X_neg1, X_cka
    else:
        return nets