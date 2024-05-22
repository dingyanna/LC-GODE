import torch
import numpy as np 
from torch import nn 

def mape(true, pred, t=-1):
    '''
    true: N_sample, T 
    pred: T, N_sample, vis_batch_size
    '''   
    return np.mean( np.abs((true[:,:t] - pred[:,:t]) / true[:,:t]) , axis=1) 
    
def rmse(true, pred, t=-1):
    '''
    true: N_sample, T
    pred: T, N_sample, vis_batch_size
    ''' 
    return np.sqrt(np.mean( np.square((true[:,:t] - pred[:,:t]) ) , axis=1)) 
    
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight.data)
            m.bias.data.fill_(0.1)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_next_batch_new(dataloader,device=torch.device('cpu')):
    data_dict = dataloader.__next__()
    #device_now = data_dict.batch.device
    return data_dict.to(device)


def normalize(x, loss=True):
    '''
     
    Input:
      - x: observed loss value
      - min_: A Boolean specifying that we expect learning to minimize 
      - l_hard, u_hard: soft lower/upper bounds for model performance 
      or maximize the performance measure.
      - l_soft, u_soft: possibly infinite lower/upper bound for loss curves 
    '''
    if loss:
        min_ = True
        l_hard = 0 
        u_hard = np.log(10)
        l_soft = 0
        u_soft = np.inf
    else:
        min_ = False
        l_hard = 0 
        u_hard = 1
        l_soft = 0
        u_soft = 1
    a, b, c, d = get_coeff(l_soft, u_soft, l_hard, u_hard)
    y = (c / (1 + np.exp(-a * (x - b))) )+ d
    return cr(min_,y)

def inverse_normalize(x, loss=True):
    '''
    return 
    ''' 
    if loss: 
        min_ = True
        l_hard = 0 
        u_hard = np.log(10)
        l_soft = 0
        u_soft = np.inf
    else:
        min_ = False
        l_hard = 0 
        u_hard = 1
        l_soft = 0
        u_soft = 1
    a, b, c, d = get_coeff(l_soft, u_soft, l_hard, u_hard)
    #return (np.log((cr(min_,x) - d) / (c - (cr(min_,x) - d)) ) - b) / a
    return b - (np.log((c / (cr(min_,x) - d)) - 1) / a)

def cr(min_, y):
    '''
    # Reflection around 0.5 (later)
    '''
    return y

def get_coeff(l_hard, u_hard, l_soft, u_soft):
    a = 2 / (u_soft - l_soft)
    b = - (u_soft + l_soft) / (u_soft - l_soft)
    c = (1 + np.exp(-a * (u_hard - b)) + np.exp(- a * (l_hard - b)) + np.exp(-a * (u_hard + l_hard - 2 * b))) / (np.exp(-a * (l_hard-b)) - np.exp(-a * (u_hard-b)))
    d = -c / (1 + np.exp(-a * (l_hard - b)))
    return a, b, c, d
 