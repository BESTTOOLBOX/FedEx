import copy
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
import torch.nn.functional as F
from torchlars import LARS
from models.encoder_head import feature_extract
from models.lstm import ModelLSTMShakespeare
import numpy as np


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class FL_client(object):
    def __init__(self, args):
        self.args = args
        if self.args.dataset == 'shakespeare':
            self.model = ModelLSTMShakespeare(args=args).to(args.device)
        elif self.args.dataset == 'reddit':
            self.model = None
        self.ep = 1
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)

    def assign_model(self, weights):
        self.model.load_state_dict(copy.deepcopy(weights))



class FL_client_text(FL_client):
    def __init__(self, args, dataset=None, idxs=None):
        super(FL_client_text, self).__init__(args)
        self.ldr_train = DataLoader(DatasetSplit(dataset, list(idxs)),
                                    batch_size=self.args.local_bs,
                                    shuffle=True,
                                    )
        self.n_sample = len(self.ldr_train)

    def local_train(self, lr, H):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        epoch_loss = []
        for iter in range(H):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):

                data, labels = data.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = self.model(data)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)


class LocalUpdate(object):
    def __init__(self, args, train_dataset=None, idxs=None):
        self.local_bs = args.local_bs
        self.optim = args.optim
        self.device = args.device
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.ldr_train = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=args.local_bs, shuffle=True, drop_last=False)

        self.ep = 1
        self.warmup_epochs = 10

        if 'CR' in args.mloss:
            self.dim_option = True
        else:
            self.dim_option = False

    def train_optim(self, net, lr):
        if self.optim == 'adam':
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=(0.9, 0.999))
        elif self.optim == 'amsgrad':
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-4,
                                         amsgrad=True)
        elif self.optim == 'sgd':
            return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                                        weight_decay=1e-4)
        elif self.optim == 'LARS':
            return LARS(net.parameters(), lr=initial_lr)

    def tensor_norm(self, feature):
        return F.normalize(feature, p=2, dim=1)


class AvgUpdate(LocalUpdate):
    def __init__(self, args, train_dataset=None, idxs=None):
        super().__init__(args, train_dataset, idxs)


    def local_train(self, net, lr, local_iter=10):
        net.train()
        # train and update
        optimizer = self.train_optim(net, lr)
        epoch_loss = []
        self.ep = local_iter
        for _ in range(self.ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                # if batch_idx == local_iter:
                #     break

                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                images.requires_grad, labels.requires_grad = False, False
                labels = labels.long()

                _, _, out = net(images)
                if labels.size(0) == 1:
                    out = out.reshape(1,-1)

                # empirical loss
                loss = self.loss_func(out, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())


            epoch_loss.append(sum(batch_loss) / len(batch_loss))


        return net, sum(epoch_loss) / len(epoch_loss)


class ProxUpdate(LocalUpdate):
    def __init__(self, args, train_dataset=None, idxs=None):
        super().__init__(args, train_dataset, idxs)

        self.mu = args.mu

    def local_train(self, net, global_net, lr, local_iter=10, flag=1):
        net.train()
        global_weight_collector = list(global_net.parameters())
        # train and update
        optimizer = self.train_optim(net, lr)
        epoch_loss = []
        self.ep = local_iter
        for _ in range(self.ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                # if batch_idx == local_iter:
                #     break

                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                images.requires_grad, labels.requires_grad = False, False
                labels = labels.long()

                _, _, out = net(images)
                if labels.size(0) == 1:
                    out = out.reshape(1,-1)

                # empirical loss
                loss = self.loss_func(out, labels)
                # for fedprox
                fed_prox_reg = 0.0
                if flag:
                    for param_index, param in enumerate(net.parameters()):
                        fed_prox_reg += ((self.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                    loss += fed_prox_reg
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())


            epoch_loss.append(sum(batch_loss) / len(batch_loss))


        return net, sum(epoch_loss) / len(epoch_loss)


class MoonUpdate(LocalUpdate):
    def __init__(self, args, train_dataset=None, idxs=None):
        super().__init__(args, train_dataset, idxs)
        self.cos = nn.CosineSimilarity(dim=-1).to(self.device)
        self.temperature = args.t
        self.mu = args.mu
        self.dim = args.dim
        self.modelName = args.models
        
    def gram_matrix(self, features):

        return features @ features.T

    def center_kernel(self, K):

        N = K.shape[0]
        one_N = np.ones((N, N)) / N
        return K - one_N @ K - K @ one_N + one_N @ K @ one_N

    def frobenius_inner_product(self, A, B):

        return np.trace(A.T @ B)

    def compute_cka(self, features_x, features_y):

        Kx = self.gram_matrix(features_x)
        Ky = self.gram_matrix(features_y)

        Kx_centered = self.center_kernel(Kx)
        Ky_centered = self.center_kernel(Ky)

        cka = self.frobenius_inner_product(Kx_centered, Ky_centered)
        cka /= np.sqrt(self.frobenius_inner_product(Kx_centered, Kx_centered) * self.frobenius_inner_product(Ky_centered, Ky_centered))
        
        return cka    
    

    def local_train(self, net, global_net, previous_nets, lr, local_iter=10, flag=1, mu=1):
        net.train()
        # train and update
        optimizer = self.train_optim(net, lr)
        epoch_loss, epoch_loss1, epoch_loss2, minibatch_loss_record = [], [], [], []
        self.ep = local_iter
        for ep in range(self.ep):
            batch_loss, batch_loss1, batch_loss2 = [], [], []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.modelName != 'squeezenet':
                    if batch_idx == 1:
                        break

                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                images.requires_grad, labels.requires_grad = False, False
                labels = labels.long()

                _, pro1, out = net(images)
                if labels.size(0) == 1:
                    pro1 = pro1.reshape(1, -1)
                _, pro2, _ = global_net(images)
                if labels.size(0) == 1:
                    pro2 = pro2.reshape(1, -1)
                
                #if ep == (self.ep - 1):   
                #    cka_value = self.compute_cka(pro1.cpu().detach().numpy(), pro2.cpu().detach().numpy())
                

                # positive sample
                posi = self.cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                # negetive sample
                if flag:
                    for prev_net in previous_nets:
                        prev_net.to(self.device)
                        _, pro3, _ = prev_net(images)
                        if labels.size(0) == 1:
                            pro3 = pro3.reshape(1, -1)
                        nega = self.cos(pro1, pro3)
                        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= self.temperature
                target = torch.zeros(images.size(0)).to(self.device).long()

                # contrastive loss
                #loss2 = mu * self.loss_func(logits, target)
                # empirical loss
                if labels.size(0) == 1:
                    out = out.reshape(1,-1)
                loss1 = self.loss_func(out, labels)
                # loss = loss1 + loss2
                loss = loss1

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_loss1.append(loss1.item())
                #batch_loss2.append(loss2.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            minibatch_loss_record.append(batch_loss)
            #epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
        ans=0
        if len(epoch_loss)==0:
            ans=0
        else:
            ans=sum(epoch_loss) / len(epoch_loss)
        return net, ans, \
               minibatch_loss_record#sum(epoch_loss1) / len(epoch_loss1), sum(epoch_loss2) / len(epoch_loss2), cka_value
               

class Moon_cls_Update(LocalUpdate):
    def __init__(self, args, train_dataset=None, idxs=None, cls_metrics=None):
        super().__init__(args, train_dataset, idxs)
        self.cos = nn.CosineSimilarity(dim=-1).to(self.device)
        self.temperature = args.t
        self.mu = args.mu
        if cls_metrics != None:
            self.metric = self.cls_metric(cls_metrics=cls_metrics, num_sample=idxs)
        self.dim = args.dim


    def cls_metric(self, cls_metrics, num_sample):
        new_cls = copy.deepcopy(cls_metrics)
        for id in range(len(cls_metrics)):
            new_cls[id] = cls_metrics[id] * (5000 / (num_sample[id] + 5000)) + 0.4
        print('class wise metric: ', new_cls)
        return new_cls

    def cls_weight(self, labels):
        local_cls = torch.zeros((self.local_bs, 1)).to(self.device).float()
        for i in range(len(labels)):
            local_cls[i, 0] = self.metric[labels[i].item()].item()
        return local_cls



    def local_train(self, net, global_net, previous_nets, lr, local_iter=10, flag=1, iter=0):
        net.train()
        # train and update
        optimizer = self.train_optim(net, lr)

        # Create the learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.warmup_epochs, T_mult=2)

        epoch_loss, epoch_loss1, epoch_loss2 = [], [], []
        self.ep = local_iter
        for epoch in range(self.ep):

            if self.optim == 'LARS' and (epoch * (iter + 1)) < self.warmup_epochs:
                cs_lr = lr * (epoch * (iter + 1) + 1) / self.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cs_lr

            batch_loss, batch_loss1, batch_loss2 = [], [], []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                # if batch_idx == local_iter:
                #     break

                images, labels = images.to(self.device), labels.to(self.device)
                images.requires_grad, labels.requires_grad = False, False
                labels = labels.long()
                optimizer.zero_grad()

                _, pro1, out = net(images)
                if self.dim_option:
                    if labels.size(0) == 1:
                        pro1 = pro1.reshape(1, -1)
                    pro1 = pro1[:, :self.dim]
                _, pro2, _ = global_net(images)
                if self.dim_option:
                    if labels.size(0) == 1:
                        pro2 = pro2.reshape(1, -1)
                    pro2 = pro2[:, :self.dim]
                # positive sample
                posi = self.cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                # negetive sample
                if flag:
                    for prev_net in previous_nets:
                        prev_net.to(self.device)
                        _, pro3, _ = prev_net(images)
                        if self.dim_option:
                            if labels.size(0) == 1:
                                pro3 = pro3.reshape(1, -1)
                            pro3 = pro3[:, :self.dim]
                        nega = self.cos(pro1, pro3)
                        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    self.temperature = self.cls_weight(labels=labels)
                logits /= self.temperature
                '''contrastive loss'''
                target = torch.zeros(images.size(0)).to(self.device).long()
                loss2 = self.mu * self.loss_func(logits, target)

                '''empirical loss'''
                if labels.size(0) == 1:
                    out = out.reshape(1,-1)
                loss1 = self.loss_func(out, labels)
                '''loss = loss1 + loss2'''
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2)

            scheduler.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))

        return net, sum(epoch_loss) / len(epoch_loss), sum(epoch_loss1) / len(epoch_loss1), sum(epoch_loss2) / len(epoch_loss2),


class L_S_Update(LocalUpdate):
    def __init__(self, args, train_dataset=None, idxs=None):
        super().__init__(args, train_dataset, idxs)

        self.dim = args.dim
        
    def gram_matrix(self, features):

        return features @ features.T

    def center_kernel(self, K):

        N = K.shape[0]
        one_N = np.ones((N, N)) / N
        return K - one_N @ K - K @ one_N + one_N @ K @ one_N

    def frobenius_inner_product(self, A, B):

        return np.trace(A.T @ B)

    def compute_cka(self, features_x, features_y):

        Kx = self.gram_matrix(features_x)
        Ky = self.gram_matrix(features_y)

        Kx_centered = self.center_kernel(Kx)
        Ky_centered = self.center_kernel(Ky)

        cka = self.frobenius_inner_product(Kx_centered, Ky_centered)
        cka /= np.sqrt(self.frobenius_inner_product(Kx_centered, Kx_centered) * self.frobenius_inner_product(Ky_centered, Ky_centered))
        
        return cka    
    


    def local_train(self, net, global_net, previous_nets, lr, local_iter=10, flag=1, mu=1):
        net.train()
        # train and update
        optimizer = self.train_optim(net, lr)
        epoch_loss, minibatch_loss_record = [], []
        self.ep = local_iter
        for ep in range(self.ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if batch_idx == 1:
                    break

                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                images.requires_grad, labels.requires_grad = False, False
                labels = labels.long()

                _, _, log_probs = net(images)

                loss = self.loss_func(log_probs, labels)


                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            minibatch_loss_record.append(batch_loss)


        return net, sum(epoch_loss) / len(epoch_loss), \
               minibatch_loss_record#sum(epoch_loss1) / len(epoch_loss1), sum(epoch_loss2) / len(epoch_loss2), cka_value


# Set the initial learning rate
initial_lr = 4.8

# Set the total number of epochs
total_epochs = 100  # Adjust this value according to your needs


# Create your optimizer (e.g., Adam, SGD, etc.)
# optimizer = optim.Adam(model.parameters(), lr=initial_lr)



'''
# Training loop
for epoch in range(total_epochs):
    # Perform warm-up
    if epoch < warmup_epochs:
        cs_lr = lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = cs_lr

    # Train your model for the current epoch

    # Adjust the learning rate
    scheduler.step()

    # Rest of the training loop...
'''

