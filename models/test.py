import torch.nn as nn
import copy
import torch
import numpy as np
from utils.init_models import nu_classes
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch.nn.functional as F


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().to(args.device)
    loss_collector = []
    classwise_count = torch.Tensor([0 for _ in range(nu_classes(args))]).to(args.device)# args.num_classes
    classwise_correct = torch.Tensor([0 for _ in range(nu_classes(args))]).to(args.device)
    y_true = []
    y_pred = []

    for idx, (data, target) in enumerate(datatest):
        if idx == 500:
            break
        data, target = data.to(args.device), target.to(args.device)
        _, _, out = net_g(data)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        loss_collector.append(loss.item())
        total += data.data.size()[0]
        correct += (pred_label == target.data).sum().item()
        # f1 score
        y_true += target.tolist()
        y_pred += pred_label.tolist()

        for class_idx in range(nu_classes(args)):# args.num_classes
            class_elem = (target == class_idx)
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += ((target == pred_label)[class_elem].sum().item())
    

    class_acc = classwise_correct / classwise_count
    test_loss = sum(loss_collector) / len(loss_collector)
    accuracy = correct / float(total)
    return {'loss': test_loss, 'acc': accuracy, 'class_acc': class_acc,
            'f1_score': f1_score(y_true, y_pred, average=None),
            'avg_f1_score': f1_score(y_true, y_pred, average='micro')}

def test_img_all(net_g, nets, datatest, args):
    net_g.eval()
    # testing
    total = 0.
    correct = 0.
    local_acc = torch.Tensor([0 for _ in range(args.num_users)]).to(args.device)
    classwise_count = torch.Tensor([0 for _ in range(args.num_classes)]).to(args.device)
    classwise_correct = torch.Tensor([0 for _ in range(args.num_classes)]).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    loss_collector = []
    y_true = []
    y_pred = []

    for idx, (data, target) in enumerate(datatest):
        if idx == 500:
            break
        data, target = data.to(args.device), target.to(args.device)
        _, _, out = net_g(data)
        _, pred_label = torch.max(out.data, 1)
        loss = criterion(out, target)
        loss_collector.append(loss.item())
        total += data.data.size()[0]
        correct += (pred_label == target.data).sum().item()
        # f1 score
        y_true += target.tolist()
        y_pred += pred_label.tolist()
        for id, user in nets.items():
            net_u = user.net
            net_u.eval()
            _, _, out_u = net_u(data)
            _, pred_label_u = torch.max(out_u.data, 1)
            local_acc[id] += (pred_label_u == target.data).sum().item()

        for class_idx in range(args.num_classes):
            class_elem = (target == class_idx)
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += ((target == pred_label)[class_elem].sum().item())

    class_acc = classwise_correct / classwise_count
    test_loss = sum(loss_collector) / len(loss_collector)
    accuracy = correct / float(total)
    local_acc = local_acc / float(total)
    return {'loss': test_loss, 'acc': accuracy, 'class_acc': class_acc,
            'local_mean': torch.mean(local_acc), 'local_var': torch.var(local_acc),
            'f1_score':f1_score(y_true, y_pred, average=None),}

def gram_matrix(features):

    return features @ features.T

def center_kernel(K):

    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    return K - one_N @ K - K @ one_N + one_N @ K @ one_N

def frobenius_inner_product(A, B):

    return np.trace(A.T @ B)

def compute_cka(features_x, features_y):

    Kx = gram_matrix(features_x)
    Ky = gram_matrix(features_y)

    Kx_centered = center_kernel(Kx)
    Ky_centered = center_kernel(Ky)

    cka = frobenius_inner_product(Kx_centered, Ky_centered)
    numerator = cka
    denominator = np.sqrt(frobenius_inner_product(Kx_centered, Kx_centered) * frobenius_inner_product(Ky_centered, Ky_centered))
    cka_normalized = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    
    return cka_normalized

def test_cka(net, net_g, datatest, args):
    net_g.eval()
    net.eval()

    for idx, (data, target) in enumerate(datatest):
        if idx == 1:
            break
        data = data.to(args.device)
        _, pro1, _ = net(data)
        _, pro2, _ = net_g(data)
        cka_value = compute_cka(pro1.cpu().detach().numpy(), pro2.cpu().detach().numpy())


    return cka_value
    
    
def test_text(net_g, datatest, args):
    net_g.eval()
    data_loader = DataLoader(datatest, batch_size=args.bs)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)

            if idx == 100:
                break

            _, _, log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        # test_loss /= len(data_loader.dataset)
        # accuracy = 100.00 * float(correct) / len(data_loader.dataset)
        test_loss /= (100 * args.bs)
        accuracy = 100.00 * float(correct) / (100 * args.bs)
        
        return accuracy, test_loss

