import copy

def GraAvg(g, freq, idxs_users):
    g_avg = copy.deepcopy(g[idxs_users[0]])
    for net_id, grad in g.items():
        mul = freq[net_id]
        if net_id == idxs_users[0]:
            for key in grad:
                g_avg[key] = grad[key] * mul
        else:
            for key in grad:
                g_avg[key] += grad[key] * mul
    return g_avg

def W_AGG(global_w, nets_this_round, freq):
    for i, (net_id, net) in enumerate(nets_this_round.items()):
        net_para = net.state_dict()
        if i == 0:
            for key in net_para:
                global_w[key] = net_para[key] * freq[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * freq[net_id]
    return global_w

def w_correction(global_w, nets_this_round):
    diff_pool = {k: None for k in range(len(nets_this_round))}
    for net_id, net in nets_this_round.items():
        net_para = net.state_dict()
        w = copy.deepcopy(net_para)
        for key in net_para:
            w[key] = global_w[key] - net_para[key]
        diff_pool[net_id] = w
    return diff_pool

def agg_func(protos, cls_weight):
    """
    Returns the average of the PROTOTYPES.
    """
    cls = 10
    glob_proto = copy.deepcopy(protos[0])
    for user_i, proto_list in protos.items():
        if user_i == 0:
            for c in range(cls):
                glob_proto[c] = proto_list[c] * cls_weight[user_i][0][c]
        else:
            for c in range(cls):
                glob_proto[c] += proto_list[c] * cls_weight[user_i][0][c]

    return glob_proto

def AGG_TE(global_w, old_w, alpha=1):
    new_w = copy.deepcopy(old_w)
    for key in global_w:
        new_w[key] = alpha * global_w[key] + (1 - alpha) * old_w[key]
    return new_w

def correction(net_para, gradient):
    # net_para = net.state_dict()
    w = copy.deepcopy(net_para)
    for key in gradient:
        w[key] = net_para[key] + gradient[key]
    return w