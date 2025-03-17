import copy
import threading

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv

from Client import MoonUpdate, Moon_cls_Update, AvgUpdate, L_S_Update
from models.Fed import W_AGG, w_correction, AGG_TE, GraAvg
from models.test import *
from utils.datasets import data_assignment
from utils.init_models import init_nets
from utils.options import args_parser
from utils.others import initial_logging, name_save


def system_utility_time(idx, T, communicaton_time, computation_time, local_iter):
    if T < (communicaton_time[idx] + computation_time[idx] * local_iter):
        return T / (communicaton_time[idx] + computation_time[idx] * local_iter)
    else:
        return 1

def bigzero(K,Sn):
    if K-Sn<=0:
        return 0
    return int(K-Sn)

def get_user_list(bs, filepath):
    device_type = []
    local_energy_origin = []
    communication_rate = []
    communication_time = []
    communication_energy = []
    computation_time = []
    computation_energy = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            device_type.append(str(row[1]))
            local_energy_origin.append(float(row[2]))
            communication_rate.append(float(row[3]))
            communication_time.append(float(row[4]))
            communication_energy.append(float(row[5]))
            computation_time.append(float(row[6]) * bs)
            computation_energy.append(float(row[7]) * bs)
    return device_type, local_energy_origin, communication_rate, communication_time, communication_energy, computation_time, computation_energy



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu))
    args.unit = 5

    logger = initial_logging()
    dirname = './log/result/result20240629/'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    

    logger.info(args.device)
    logger.info(">> Total communication rounds: " + str(args.epochs))
    names = name_save(args)
    logger.info('>>' + names['file_name'])

    INF = 2147483647
    bs = args.local_bs
    

    
    if args.dataset == 'mnist':
        filepath = "./device_data/userList_cnn_mnist.csv"
    elif args.dataset == 'cifar':
        filepath = "./device_data/userList_squeezenet_cifar.csv"
    elif args.dataset == 'har':
        filepath = "./device_data/userList_cnn_har.csv"
    elif args.dataset == 'shakespeare':
        filepath = "./device_data/userList_lstm_shakespeare.csv"
    elif args.dataset == 'tsrd' and args.models == 'mobilenet':
        filepath = "./device_data/userList_mobilenet_tsrd.csv"
    elif args.dataset == 'tsrd' and args.models == 'mobileformer':
        filepath = "./device_data/userList_mobileformer_tsrd.csv"
    else:
        exit('Error:unrecognized dataset or model')
    
    # record initialization
    local_age = [1 for nn in range(0, args.num_users)]
    local_time = [0 for nn in range(0, args.num_users)]
    local_loss = [0 for nn in range(0, args.num_users)]
    overlap_loss = [0 for nn in range(0, args.num_users)]
    local_last_sn = {k: 0 for k in range(0, args.num_users)}
    local_last_k = [args.local_iter for nn in range(0, args.num_users)]
    local_iter = [args.local_iter for nn in range(0, args.num_users)]
    local_acc_iter = [0 for nn in range(0, args.num_users)]
    local_cka_records = [0 for nn in range(0, args.num_users)]
    cka_matrix = np.zeros((args.num_users, args.num_users+1))
    device_type, local_energy_origin, communication_rate, communication_time, communication_energy, computation_time, computation_energy = get_user_list(
        bs, filepath)

    for idx in range(args.num_users):
        local_time[idx] = communication_time[idx] + local_iter[idx] * computation_time[idx]
    print(local_time)

    # load dataset and split users
    logger.info(">> Partitioning data")
    dataset_train, dataset_test, dict_users, freq_user = data_assignment(args)

    logger.info(freq_user)

    # build models
    logger.info(">> Initializing nets")
    nets = init_nets(n_parties=args.num_users, args=args)

    # global models initialization
    global_models = init_nets(1, args)
    global_model = global_models[0]
    global_model.eval()
    for param in global_model.parameters():
        param.requires_grad = False
    global_w = global_model.state_dict()

    w_init = copy.deepcopy(global_w)
    
    assist_glob_model = init_nets(1, args)
    assist_glob_model = assist_glob_model[0]
    assist_glob_model.eval()
    for param in assist_glob_model.parameters():
        param.requires_grad = False

    # training initialization
    lr = args.lr
    X = np.zeros((args.epochs, 6))
    arr = np.empty((0, 10), dtype=np.float32)
    arr_div_feature = np.empty((0, 10), dtype=np.float32)

    
    grad_records = {g_i: None for g_i in range(args.num_users)}
    grad_records_overlap = {g_i: None for g_i in range(args.num_users)}



    T = args.T
    flag = 0
    totalTime = 0
    m = max(int(args.frac * args.num_users), 1)
    overlap_flag = 0
    overlap_iter = 0
    max_round_time = 0
    
    for idx in range(args.num_users):
        max_round_time = max(max_round_time, communication_time[idx] + computation_time[idx] * args.local_iter)
    print('max round time: ', max_round_time)

    # training
    for iter in range(args.epochs):
        logger.info(">> in comm round:" + str(iter))
#        if iter > 150:
#            lr = lr / 1.1

        # user selection
        if iter <= 1:
            idxs_users = [nn for nn in range(0, args.num_users)]
            local_util = []
            system_utility_time_record = []
            system_utility_loss_record = []

            
        else:
#            overlap_flag = 1###################
#            overlap_iter = 1###################
            system_utility_time_record = [
                system_utility_time(nn, T, communication_time, computation_time, local_iter[nn]) for nn in
                range(0, args.num_users)]
            system_utility_loss_record = [((len(dict_users[nn]) * local_loss[nn]) ** 0.5) for nn in
                                          range(0, args.num_users)]
            if overlap_flag == 0:
                local_util = [
                (system_utility_loss_record[nn] ** args.a) * (system_utility_time_record[nn] ** args.b) #+ (0.1 * float(math.log(iter) / L_i[nn])) ** 0.5
                for nn in range(0, args.num_users)]
            else:
                local_util = [
                    (system_utility_loss_record[nn] ** args.a) * 
                    ((1 / (computation_time[nn] * (local_last_k[nn] - local_last_sn[nn]) + communication_time[nn])) ** args.b) #* ((local_age[nn] / (local_age[nn] + 1)) ** args.delta) 
                    for nn in range(0, args.num_users)]
                #print(local_util)    


            # Forced selection    
            for nn in range(0, args.num_users):
                if local_age[nn] > 20:
                    local_util[nn] = INF
            #print(local_util)
            
            temp_local_util = copy.deepcopy(local_util)
            idxs_users = []
            for ii in range(m):
                idxs_users.append(temp_local_util.index(max(temp_local_util)))
                temp_local_util[temp_local_util.index(max(temp_local_util))] = -INF


        logger.info(">> The selection user in round " + str(iter))
        logger.info(idxs_users)


        nets_this_round = {k: nets[k] for k in idxs_users}
        extra_local_iters = {k: 0 for k in idxs_users}
        
        upload_this_round = {k: nets[k] for k in idxs_users}

        
        if overlap_flag == 0:
            for net in nets.values():
                net.load_state_dict(global_w)
            logger.info(">> Initialize from the same models")
            

        # time of this round
        tempTime = 0
        timerecord = []
        for idx in idxs_users:
            if overlap_flag:
                #tempTime = max(tempTime, communication_time[idx] + computation_time[idx] * (local_last_k[idx] - local_last_sn[idx]))
                tempTime = max(tempTime, communication_time[idx] + computation_time[idx] * max(local_last_k[idx] - local_last_sn[idx], 0))
                timerecord.append(communication_time[idx] + computation_time[idx] * max(local_last_k[idx] - local_last_sn[idx], 0))
                #timerecord.append(communication_time[idx] + computation_time[idx] * (local_last_k[idx] - local_last_sn[idx]))
            else:
                tempTime = max(tempTime, communication_time[idx] + computation_time[idx] * args.local_iter)
        print(f'iter:{iter}')
        print(f'tempTime:{tempTime}')
        print(timerecord)
        
        for idx in idxs_users:
            local_time[idx] = (local_last_k[idx] - local_last_sn[idx]) * computation_time[idx]
            extra_local_iters[idx] = int((tempTime - local_time[idx]) // computation_time[idx])
            
#            if extra_local_iters[idx] > local_iter[idx]:
#                extra_local_iters[idx] = int(local_iter[idx])
            if extra_local_iters[idx] > int(args.setK):
                extra_local_iters[idx] = int(args.setK)


        logger.info(">> Local training starts!")
        
        # Local training
        for idx, net in nets_this_round.items():
            
            prev_model = []


            if 'cls' in args.mloss:
                local = Moon_cls_Update(args=args, train_dataset=dataset_train, idxs=dict_users[idx])
            elif args.dataset == 'shakespeare':
                local = L_S_Update(args=args, train_dataset=dataset_train, idxs=dict_users[idx])
            else:
                local = MoonUpdate(args=args, train_dataset=dataset_train, idxs=dict_users[idx])
            
            if overlap_flag:
                # k-Sn
                if local_last_sn[idx] < int(local_last_k[idx]):
                    
                    preserved_net = copy.deepcopy(net)
                    
                    if iter == overlap_iter + 1:
                        net.load_state_dict(global_w)
                                
                    elif (iter >= overlap_iter + 1) and (local_age[idx] <= 2):
                        start_w = copy.deepcopy(net.state_dict())
                        correction = copy.deepcopy(grad_records[idx])
                        if correction is None:
                            net.load_state_dict(global_w)
                        else:
                            for j in correction:
                                start_w[j] = start_w[j] + correction[j] - acc_global_g[j]

                            net.load_state_dict(start_w)
                        
                    elif (iter >= overlap_iter + 1) and (local_age[idx] > 2):
                        net.load_state_dict(global_w)

                        
                    prev_w = copy.deepcopy(net.state_dict())
                    
                    local_net, loss, minibatch_loss_record = local.local_train(net=net.to(args.device), 
                                                            local_iter=int(local_last_k[idx]-local_last_sn[idx]),
                                                            lr=lr, flag=flag,
                                                            global_net=assist_glob_model,
                                                            previous_nets=prev_model)
                    net = local_net
                    local_acc_iter[idx] += int(local_last_k[idx]-local_last_sn[idx])
                    
                    #record gradient in this phase
                    if iter != 0:
                        
                        local_w = local_net.state_dict()
                        # accumulated gradients per local updates
                        g = copy.deepcopy(local_w)
                        for k in g:
                            if grad_records_overlap[idx] is None:
                                g[k] = prev_w[k] - local_w[k]
                            else:
                                g[k] = prev_w[k] - local_w[k] + grad_records_overlap[idx][k]
                        grad_records[idx] = g
                    
                        
                    upload_this_round[idx] = copy.deepcopy(local_net)
                    local_loss[idx] = loss
                    logger.info('client %d Local Training Loss: %f' % (idx, loss))
                    



                else:
                    # immediately upload
                    grad_records[idx] = copy.deepcopy(grad_records_overlap[idx])
                    upload_this_round[idx] = copy.deepcopy(net)
                    
                    if local_age[idx] <= 2:
                        start_w = copy.deepcopy(net.state_dict())
                        correction = copy.deepcopy(grad_records[idx])
                    
                        for j in correction:
                            start_w[j] = start_w[j] + correction[j] - acc_global_g[j]

                        net.load_state_dict(start_w)
                        
                    else:
                        net.load_state_dict(global_w)
                        
                
                    local_loss[idx] = overlap_loss[idx]
                    logger.info('client %d immediately upload Loss: %f' % (idx, local_loss[idx]))
                    
                
                
                # overlap
                if (iter >= overlap_iter + 1) and (extra_local_iters[idx] >= 1):
                    prev_w = copy.deepcopy(net.state_dict())
                    local_net, loss, _ = local.local_train(net=net.to(args.device), 
                                                                    local_iter=extra_local_iters[idx], lr=lr, flag=flag,
                                                                    global_net=assist_glob_model,
                                                                    previous_nets=prev_model)
                    
                    net = local_net
                    local_acc_iter[idx] += extra_local_iters[idx]
                    
                    local_w = local_net.state_dict()
                    # accumulated gradients per local updates
                    g_overlap = copy.deepcopy(local_w)
                    for k in g_overlap:
                        g_overlap[k] = prev_w[k] - local_w[k]
                    grad_records_overlap[idx] = g_overlap
                    overlap_loss[idx] = loss
                    logger.info('client %d Local Overlap Training Loss: %f' % (idx, loss))
                
                elif (iter >= 1) and (extra_local_iters[idx] == 0):
                    grad_records_overlap[idx] = None
            else:
                
                local_net, loss, _ = local.local_train(net=net.to(args.device),
                                                              local_iter=args.local_iter, lr=lr, flag=flag,
                                                              global_net=assist_glob_model,
                                                              previous_nets=prev_model)            
                net = local_net
                local_loss[idx] = loss
                local_acc_iter[idx] += args.local_iter
                logger.info('client %d Local oort Training Loss: %f' % (idx, loss))


        logger.info(">> Local training completes!")
        
        # update last Sn and last K
        if iter >= 1 and overlap_flag:
            for idx in extra_local_iters:
                local_last_sn[idx] = extra_local_iters[idx]
            for idx in idxs_users:
                local_last_k[idx] = local_iter[idx]
                
        # update age
        for ii in range(0, args.num_users):
                if ii in idxs_users:
                    local_age[ii] = 1
                else:
                    local_age[ii] = local_age[ii] + 1
        # update time
        totalTime += tempTime
        
        total_data_points = sum([len(dict_users[r]) for r in idxs_users])
        fed_avg_freqs = {r: len(dict_users[r]) / total_data_points for r in idxs_users}
        
        # update cka
        if overlap_flag == 0:
            
            for idx, net in nets_this_round.items():
                test_dataloader = DataLoader(dataset_test, batch_size=64)
                local_cka_records[idx] = test_cka(net, assist_glob_model, test_dataloader, args)
                               
            cka_mean = sum(local_cka_records) / len(local_cka_records)
            logger.info('CKA mean: %f' % (cka_mean))
            if cka_mean >= 0.7 and iter > 10:
                overlap_flag = 1
                overlap_iter = iter
                logger.info('-------------------overlap start iter: %f------------------------' % (iter + 1))
                
        else:
            for idx, net in upload_this_round.items():
                test_dataloader = DataLoader(dataset_test, batch_size=64)
                local_cka_records[idx] = test_cka(net, assist_glob_model, test_dataloader, args)
                logger.info('client %d CKA: %f' % (idx, local_cka_records[idx]))
        
        
        w_pre = copy.deepcopy(global_w)

        # update global model
        if overlap_flag:
            global_w = W_AGG(w_pre, upload_this_round, fed_avg_freqs)
        else:
            global_w = W_AGG(w_pre, nets_this_round, fed_avg_freqs)
        global_model.load_state_dict(global_w)

        
        # update average gradient
        grad_records_this_round = {k: grad_records[k] for k in idxs_users}
        if iter > overlap_iter and overlap_flag:
            acc_global_g = GraAvg(grad_records_this_round, fed_avg_freqs, idxs_users)
       


        '''Temperal ensemble'''
        assist_w = AGG_TE(global_w=global_w, old_w=w_pre, alpha=0.99)
        assist_glob_model.load_state_dict(assist_w)


        if args.dataset == 'shakespeare':
            acc_test, loss_test_1 = test_text(assist_glob_model, dataset_test, args)
            print('Iter', iter)
            if overlap_flag == 0:
                print('cka mean', cka_mean)
            print("Testing accuracy: {:.2f}".format(acc_test))
            print("Testing Loss: {:.2f}".format(loss_test_1))
            logger.info("Testing accuracy: {:.2f}".format(acc_test))
            logger.info("Testing Loss: {:.5f}".format(loss_test_1))
        else:
            test_dataloader = DataLoader(dataset_test, batch_size=args.bs)
            pref = test_img(assist_glob_model, test_dataloader, args)
            logger.info("Testing accuracy: {:.2f}".format(pref['acc']))
            logger.info("Testing Loss: {:.5f}".format(pref['loss']))
            test_dataloader = DataLoader(dataset_train, batch_size=args.bs)
            pref_train = test_img(assist_glob_model, test_dataloader, args)
            print('Iter ', iter)

            print('Test acc of original glob_model: ', pref['acc'])
            

        
            
        logger.info(">> Local accumulated iters:")
        logger.info(local_acc_iter)
        
        #output results
        
        if args.dataset == 'mnist':
            logpath = dirname + "cnn_mnist" + "_alpha" + str(int(args.b)) + "_setK" + str(int(args.setK)) + "_iid" + str(int(args.iid)) + "_beta" + str(args.beta) + "_seed" + str(args.seed) + ".csv"
        elif args.dataset == 'cifar':
            logpath = dirname + "squeezenet_cifar"+ "_alpha" + str(int(args.b)) + "_setK" + str(int(args.setK))  + "_iid" + str(int(args.iid)) + ".csv"
        elif args.dataset == 'har':
            logpath = dirname + "har_cnn" + "_alpha" + str(int(args.b)) + "_setK" + str(int(args.setK)) + ".csv"
        elif args.dataset == 'shakespeare':
            logpath = dirname + "lstm_shakespeare" + "_alpha" + str(int(args.b)) + "_setK" + str(int(args.setK)) + "_seed" + str(args.seed) + ".csv"
        elif args.dataset == 'tsrd' and args.models == 'mobilenet':
            logpath = dirname + "mobilenet_tsrd.csv"
        elif args.dataset == 'tsrd' and args.models == 'mobileformer':
            logpath = dirname + "mobilefomer_tsrd.csv"
        else:
            pass
            
        record_log = True
        if record_log:
            with open(logpath, "a") as f:
                csvwriter = csv.writer(f)
                if iter == 0:
                    csvwriter.writerow(['totalTime', 'acc', 'loss'])
                if args.dataset == 'shakespeare':
                    csvwriter.writerow([totalTime, acc_test, loss_test_1])
                else:
                    csvwriter.writerow([totalTime, pref['acc'], pref['loss']])



    if args.dataset == 'shakespeare':
        acc_test, loss_test_1 = test_text(assist_glob_model, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        print("Testing Loss: {:.2f}".format(loss_test_1))
    else:

        test_dataloader = DataLoader(dataset_test, batch_size=args.bs)
        pref = test_img(global_model, test_dataloader, args)
        print("Testing accuracy: {:.2f}".format(pref['acc']))
        print("Testing Loss: {:.5f}".format(pref['loss']))
