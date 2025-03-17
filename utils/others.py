import pandas as pd
from utils.options import args_parser
import os
import logging
import datetime
import numpy as np
import csv


args = args_parser()

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def name_save(args):
    if args.iid:
        s = '{}_iid'.format(args.mloss)
    else:
        if args.imb:
            s = '{}_m{}'.format(args.mloss, args.main_class_number)
        else:
            s = "{}_p{}".format(args.mloss, args.beta)
    s_name = s + '_{}_{}_B{}_E{}_H{}_N{}_{}LR{}_{}'.format(args.models, args.dataset, args.local_bs, args.local_ep,
                                                           args.local_iter, args.num_users,
                                                           args.lr_sche, args.lr, args.optim)

    if 'overlap' in args.mloss or args.mloss == 'DGA':
        s_name = s_name + '_D{}'.format(args.delay)
    if 'moon' in args.mloss or 'CR' in args.mloss:
        s_name += '_t{}'.format(args.t)
    if args.mloss != 'DGA' or args.mloss != 'avg':
        s_name += '_mu{}'.format(args.mu)

    s_acc =  s_name + '.csv'
    s2 = 'Align_' + s_name + '.csv'
    s3 = 'Diver_' + s_name + '.csv'
    s4 = 'CKA_' + s_name + '.csv'
    s5 = 'Neg_' + s_name + '.csv'
    return {'file_name': s_name, 'acc_name': s_acc, 'align_name': s2,
            'diver_name': s3, 'CKA_name': s4, 'Neg_metric_name': s5,}


def metric_writing(name, ng, ng1, ng2):
    f = open('./data/'+ name +'_NG.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(ng)
    f.close()
    f1 = open('./data/'+ name +'_posi.csv', 'a', encoding='utf-8', newline='')
    wr1 = csv.writer(f1)
    wr1.writerow(ng1)
    f1.close()
    f2 = open('./data/'+ name +'_nega.csv', 'a', encoding='utf-8', newline='')
    wr2 = csv.writer(f2)
    wr2.writerow(ng2)
    f2.close()

def initial_logging():
    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    mkdirs(args.logdir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')
    return logger


def assign_model(model, weight):
    model.load_state_dict(weight)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def final_result_collection(X, X_align, X_neg, X_neg1, final_cka1):
    s_name, s_acc, s_ali, s_div, s_cka, s_neg = name_save(args)
    pd_data1 = pd.DataFrame(X)
    pd_data1.to_csv(s_acc)

    # x_ali = np.hstack((X_align, X_align1))
    pd_data2 = pd.DataFrame(X_align)
    pd_data2.to_csv(s_ali)

    x_n = np.hstack((X_neg, X_neg1))
    pd_data3 = pd.DataFrame(x_n)
    pd_data3.to_csv(s_neg)

    X_cka = np.zeros((args.num_users, args.num_users + 1))
    for j_id1, cka in final_cka1.items():
        cka = np.array(cka)
        for c_id in range(args.num_users + 1):
            temp_meam = cka[:, c_id]
            cka_mean = sum(temp_meam) / len(temp_meam)
            X_cka[j_id1, c_id] = cka_mean
    pd_data4 = pd.DataFrame(X_cka)
    pd_data4.to_csv(s_cka)

    # X_diver = p_diver(s_div, global_w, nets)
    # pd_data3 = pd.DataFrame(X_diver)
    # pd_data3.to_csv(s_div)

