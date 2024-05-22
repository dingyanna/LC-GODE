# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Latent SDE fit to a single time series with uncertainty quantification."""
import argparse
import math
import logging
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim
import torch.nn.functional as F
from util_tmp import * 
# from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
import torchsde
from torchdiffeq import odeint_adjoint, odeint
import time 
import calendar 
import datetime 
from seq_encoder import *
today =str(datetime.datetime.now()).split(' ')[0] 
 
# w/ underscore -> numpy; w/o underscore -> torch.
Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'train_ys', 'train_ys_obs', 'test_ys', 'test_ys_obs', 'num_train_batch', 'num_test_batch', 'min_', 'max_'])
 
class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class LatentODE(torchsde.SDEIto):

    def __init__(self, dim_in=2, dim_hid=8, theta=1.0, mu=0.1, sigma=0.5):
        super(LatentODE, self).__init__(noise_type="diagonal")
        logvar = math.log(sigma ** 2 / (2. * theta))

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        ############################
        # self.net = nn.Sequential(
        #     nn.Linear(dim_hid , 200),
        #     nn.GELU(),#tanh
        #     nn.Linear(200, 200),
        #     nn.GELU(),
        #     nn.Linear(200, dim_hid) 
        # )
        self.net = nn.Sequential(
            nn.Linear(dim_hid , dim_hid*4),
            nn.GELU(),#tanh
            nn.Linear(dim_hid*4, dim_hid), 
        )
        ############################
        # for param in self.net.parameters():
        #     param.requires_grad = False
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        #self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        #self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

        self.encoder = SelfAttentionNetwork(dim_in, dim_hid*2)
        #self.encoder = AutoRegressiveGRU(dim_in, dim_hid, dim_hid*2)
        #self.encoder = Encoder_GRU(dim_in, dim_hid, dim_hid*2)
        self.decoder = nn.Sequential(
            nn.Linear(dim_hid, dim_hid//2),
            nn.GELU(),
            nn.Linear(dim_hid//2, 1) 
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(dim_hid, 1), 
        # )
        
        init_weights(self)

    def f(self, t, y):  # Approximate posterior drift.
        '''
        y: N_sample x vis_batch_size x d_hid?
        '''
         
        # if t.dim() == 0:
        #     t = torch.full_like(y, fill_value=t)
        # # Positional encoding in transformers for time-inhomogeneous posterior.
        # return self.net(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
        if t.dim() == 0:
            t_lis = torch.tensor([t]).to(y).repeat(y.shape[0]).reshape(-1,1)
        else:
            t_lis = t
        # print('t_lis', t_lis.shape)
        # print('cat', torch.cat((t_lis,y), dim=-1).shape)
        res = self.net(y)
        #print(res.shape, '\n')
        ################## !!!!!!!!!! the result of net should have the same shape as y!!!!!!!
        return  res
        
     
    def forward(self, observed, ts, batch_size, eps=None):
          
        self.qy0_mean, self.qy0_logvar = self.encoder(observed)
        
        eps = torch.randn(1, batch_size, 1).to(observed) if eps is None else eps
        y0 = self.qy0_mean.unsqueeze(1) + eps * self.qy0_std.unsqueeze(1)
        # self.qy0_mean.unsqueeze(1).repeat(1,batch_size,1)# 
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        # original: 1x1
        # new: N_sample x 1
        logqp0 = distributions.kl_divergence(qy0, py0)  # (5,4)  #  KL(t=0).
         
        logqp0 = logqp0.sum(1) # this is summing over what dimension? hidden dimension? previously, the shape is 1 x 1, for 1 trajectory and 1 hidden units
        aug_y0 = y0
        N_sample, vis_batch_size, _ = aug_y0.shape

        ############# 
        aug_y0 = aug_y0.reshape(N_sample*vis_batch_size, -1)
        aug_ys = odeint_fn(
            self.f,
            aug_y0,
            ts,  
            rtol=args.rtol,
            atol=args.atol, 
        ) # T x (N_sample*vis_batch_size) x (d_hid+1)
        #print('aug_ys.shape', aug_ys.shape) # aug is required because of logqp_path
        
        ys  = aug_ys  
        #############  
         
        ys = self.decoder(ys) ###### decoder should only be applied to states
        
        #ys = ys.mean(-1, keepdim=True)
        #ys, logqp_path = aug_ys , aug_ys[-1, :, 1] 
        ys = ys.reshape(-1, N_sample, vis_batch_size) 
        return ys, logqp0.mean()

    def sample_p(self, N_sample, d_hid, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean.reshape(1,1,1).repeat(N_sample, 1, d_hid) + eps.unsqueeze(0) * self.py0_std.reshape(1,1,1).repeat(N_sample, 1, d_hid)
         
        y0 = y0.reshape(y0.shape[0]*y0.shape[1], -1) # (N_sample x vis_batch_size) x d_hid
        res = odeint_fn(self.f, y0, ts, method='rk4' )
         
        return res

    def sample_q(self, observed, ts, batch_size, eps=None, bm=None):
        # observed: 
   
        self.qy0_mean, self.qy0_logvar = self.encoder(observed)
         
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps 
         
        y0 = self.qy0_mean.unsqueeze(1) + eps.unsqueeze(0) * self.qy0_std.unsqueeze(1) 
        # qy0_mean: N_sample x d_hid, eps: 1 x vis_batch_size x 1
        # expect: N_sample x 1 x d_hid, eps: 1 x vis_batch_size x 1 -> N_sample x vis_batch_size x d_hid

        # ValueError: `y0` must be a 2-dimensional tensor of shape (batch, channels).
        # need to reshape 
        # remember to map back to 3-d
        N_sample = y0.shape[0]
        y0 = y0.reshape(y0.shape[0]*y0.shape[1], -1) # (N_sample x vis_batch_size) x d_hid
         
        res = odeint_fn(self.f, y0, ts, method='rk4') 
         
        res = self.decoder(res) # T x ( N_sample*vis_batch_size )
        #res = res.mean(-1, keepdim=True)
        res = res.reshape(-1, N_sample, batch_size) # [T, N_sample, vis_batch_size]
        return res

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)

class LatentSDE(torchsde.SDEIto):

    def __init__(self, dim_in=2, dim_hid=8, theta=1.0, mu=0.1, sigma=0.5):
        super(LatentSDE, self).__init__(noise_type="diagonal")
        logvar = math.log(sigma ** 2 / (2. * theta))

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        ############################
        # self.net = nn.Sequential(
        #     nn.Linear(dim_hid , 200),
        #     nn.GELU(),#tanh
        #     nn.Linear(200, 200),
        #     nn.GELU(),
        #     nn.Linear(200, dim_hid) 
        # )
        self.net = nn.Sequential(
            nn.Linear(dim_hid , dim_hid*4),
            nn.GELU(),#tanh
            nn.Linear(dim_hid*4, dim_hid), 
        )
        ############################
        # for param in self.net.parameters():
        #     param.requires_grad = False
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        #self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        #self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

        self.encoder = SelfAttentionNetwork(dim_in, dim_hid*2)
        #self.encoder = AutoRegressiveGRU(dim_in, dim_hid, dim_hid*2)
        #self.encoder = Encoder_GRU(dim_in, dim_hid, dim_hid*2)
        self.decoder = nn.Sequential(
            nn.Linear(dim_hid, dim_hid//2),
            nn.GELU(),
            nn.Linear(dim_hid//2, 1) 
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(dim_hid, 1), 
        # )

    def f(self, t, y):  # Approximate posterior drift.
        '''
        y: N_sample x vis_batch_size x d_hid?
        '''
         
        # if t.dim() == 0:
        #     t = torch.full_like(y, fill_value=t)
        # # Positional encoding in transformers for time-inhomogeneous posterior.
        # return self.net(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
        if t.dim() == 0:
            t_lis = torch.tensor([t]).to(y).repeat(y.shape[0]).reshape(-1,1)
        else:
            t_lis = t
        # print('t_lis', t_lis.shape)
        # print('cat', torch.cat((t_lis,y), dim=-1).shape)
        #res = self.net(torch.cat((t_lis,y), dim=-1))
        res = self.net(y)
        #print(res.shape, '\n')
        ################## !!!!!!!!!! the result of net should have the same shape as y!!!!!!!
        return  res
    def g(self, t, y):  # Shared diffusion. 
        res = self.sigma.repeat(y.size(0), y.shape[-1])
        #print('in g()', res.shape)
        return res # y.shape[-1]:  should have same size as y
 
    def h(self, t, y):  # Prior drift.
        return self.theta * (self.mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
         
        y = y[:, :-1]  
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True) 
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        
        y = y[:, :-1]
        g = self.g(t, y) 
        g_logqp = torch.zeros_like(y[:, -1:]) #  need to have the same shape
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, observed, ts, batch_size, eps=None):
          
        self.qy0_mean, self.qy0_logvar = self.encoder(observed)
        
        eps = torch.randn(1, batch_size, 1).to(observed) if eps is None else eps
        y0 = self.qy0_mean.unsqueeze(1) + eps * self.qy0_std.unsqueeze(1)
         
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        # original: 1x1
        # new: N_sample x 1
        logqp0 = distributions.kl_divergence(qy0, py0)  # (5,4)  #  KL(t=0).
         
        logqp0 = logqp0.sum(1) # this is summing over what dimension? hidden dimension? previously, the shape is 1 x 1, for 1 trajectory and 1 hidden units
        aug_y0 = torch.cat([y0, torch.zeros(y0.shape[0], batch_size, 1).to(y0)], dim=-1)
        N_sample, vis_batch_size, _ = aug_y0.shape
         
        aug_y0 = aug_y0.reshape(N_sample*vis_batch_size, -1)
        aug_ys = sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method=args.method,
            dt=args.dt,
            adaptive=args.adaptive,
            rtol=args.rtol,
            atol=args.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        ) # T x (N_sample*vis_batch_size) x (d_hid+1)
        #print('aug_ys.shape', aug_ys.shape) # aug is required because of logqp_path
         
        ys, logqp_path = aug_ys[:, :, 0:-1], aug_ys[-1, :, -1:] # [N_sample*vis_batch_size, 1]
        # ys: [T, N_sample*vis_batch_size, d_hid]
        # logqp_path: [N_sample*vis_batch_size x 1]
        ys = self.decoder(ys) ###### decoder should only be applied to states
        #ys, logqp_path = aug_ys , aug_ys[-1, :, 1] 
        ys = ys.reshape(-1, N_sample, vis_batch_size) 
        logqp = (logqp0.unsqueeze(-1) + logqp_path.reshape(N_sample,vis_batch_size)).mean(dim=0).mean()  # [N_sample x vis_batch_size] -> [1,1] # take average along the TIME dimension # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, N_sample, d_hid, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean.reshape(1,1,1).repeat(N_sample, 1, d_hid) + eps.unsqueeze(0) * self.py0_std.reshape(1,1,1).repeat(N_sample, 1, d_hid)
         
        y0 = y0.reshape(y0.shape[0]*y0.shape[1], -1) # (N_sample x vis_batch_size) x d_hid
        res = sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=args.dt, names={'drift': 'h'})
         
        return res

    def sample_q(self, observed, ts, batch_size, eps=None, bm=None):
        # observed: 
   
        self.qy0_mean, self.qy0_logvar = self.encoder(observed)
         
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps 
        y0 = self.qy0_mean.unsqueeze(1) + eps.unsqueeze(0) * self.qy0_std.unsqueeze(1) 
        # qy0_mean: N_sample x d_hid, eps: 1 x vis_batch_size x 1
        # expect: N_sample x 1 x d_hid, eps: 1 x vis_batch_size x 1 -> N_sample x vis_batch_size x d_hid

        # ValueError: `y0` must be a 2-dimensional tensor of shape (batch, channels).
        # need to reshape 
        # remember to map back to 3-d
        N_sample = y0.shape[0]
        y0 = y0.reshape(y0.shape[0]*y0.shape[1], -1) # (N_sample x vis_batch_size) x d_hid
        res = sdeint_fn(self, y0, ts, bm=bm, method='srk', dt=args.dt) 
        res = self.decoder(res) # T x ( N_sample*vis_batch_size )
        res = res.reshape(-1, N_sample, batch_size) # [T, N_sample, vis_batch_size]
        return res

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)

 
   
def make_lc():
    lc = np.load(f"./data/{args.source_data}_test_{args.curve_type}.npy") # [:50]
    #idx = np.random.permutation(lc.shape[0])
    #lc = lc[idx]
    
    ############################### # old data
    # ds = f"{args.source_data}_1"
    # ids = range(50) # old data
    # lc = []   
    # for i in ids:  
    #     lc.append( (np.load(f'/home/dingy6/nn_project/LC-ODE/mlp/{ds}/val_loss{i}.npy')) ) # old data
    # lc = np.stack(lc )  
    ###############################
    #lc = lc[idx ]
    #lc = np.load("car")
    #lc = normalize(lc )  
     
    args.mean = float(np.mean(lc.mean(0)))
    args.std  =  float( (lc[args.cond_len-1].std(0)))
    min_ = np.min(lc)
    max_ = np.max(lc)
      
    print(args.mean, args.std) 
    ###################################
    # Reason for unsqueezing: for observation window, 
    # we need this dimension for the feature; 
    # during prediction, we need this axis to match the last dimension of zs 
    # (T, N_sample, vis_batch_size)
    lc = lc.reshape(lc.shape[0], lc.shape[1], 1)
    ts_total = np.sort(np.random.uniform(low=1e-5, high=0.9, size=lc.shape[1] ))
    
    ts_ = ts_total[args.cond_len:]
    
    # this is for penalization of boundaries that is inherited from the original latent_sde
    ts_ext_ = ts_ # np.array([0.] + list(ts_) + [1.0])  
 
    ts_vis_ =  ts_total # np.linspace(0., 1.0, 300)
    ts_obs = ts_total[:args.cond_len].reshape(-1,1)  # T_obs x 1
    ts_obs = np.asarray([ts_obs for i in range(lc.shape[0])])
    ys_obs_ = np.concatenate((lc[:,:args.cond_len], ts_obs), axis=-1)   # N_sample x T_obs x 2
     
    ys_ = lc[:,args.cond_len:]    # T x 1 -> N_sample x T_pred x 1
     
    ts = torch.tensor(ts_).float().to(device)
    ts_ext = torch.tensor(ts_ext_).float().to(device)
    ts_vis = torch.tensor(ts_vis_).float().to(device)
    ys = torch.tensor(ys_).float() 
    ys_obs = torch.tensor(ys_obs_).float() 

    #ys = shift(ys, ys_obs[:,-1,0])

    # ys1 = shift_inverse(ys, ys_obs[:,-1,0])
    # print(torch.mean(torch.abs(ys-ys1)))
    # exit()


    train_id = range(int(args.train_test_ratio * lc.shape[0]))  
    test_id = [i for i in range(lc.shape[0]) if i not in train_id]

    # if lc.shape[0] == 10:
    #     args.train_batch_size = 2# 25 # math.gcd(len(train_id), len(test_id))
    # elif lc.shape[0] == 100:
    #     args.train_batch_size = 20
    # else:
    #     args.train_batch_size = 128

    if args.data == 'mlp':
        args.train_batch_size = 50
    else:
        args.train_batch_size = 500

    #args.train_batch_size = 20
    print('\ntrain_batch_size', args.train_batch_size, '\n')
    
    #assert len(train_id) % args.train_batch_size  == 0
    #assert len(test_id) % args.train_batch_size  == 0

    train_ys_obs = Loader(ys_obs[train_id], batch_size=args.train_batch_size, shuffle=False)
    train_ys = Loader(ys[train_id], batch_size=args.train_batch_size, shuffle=False)  # num_graph*num_ball [tt,vals,masks]
    test_ys_obs = Loader(ys_obs[test_id], batch_size=args.train_batch_size, shuffle=False)
    test_ys = Loader(ys[test_id], batch_size=args.train_batch_size, shuffle=False)  # num_graph*num_ball [tt,vals,masks]
    num_train_batch = len(train_ys_obs)
    num_test_batch = len(test_ys_obs )
    train_ys_obs = inf_generator(train_ys_obs)
    #train_graph_data_loader   = inf_generator(train_graph_data_loader)
    train_ys = inf_generator(train_ys)
    test_ys_obs  = inf_generator(test_ys_obs)
    #test_graph_data_loader    = inf_generator(test_graph_data_loader)
    test_ys  = inf_generator(test_ys) 

    return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, train_ys, train_ys_obs, test_ys, test_ys_obs, num_train_batch, num_test_batch, min_, max_ )


def make_data():
    data_constructor =make_lc 
    # { 
    #     'cnn': make_lc, 
    # }[args.data]
    return data_constructor()

def test(global_step, model):
    ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, train_ys, train_ys_obs, test_ys, test_ys_obs, num_train_batch, num_test_batch, min_, max_ = make_data()
     
     
    checkpt = torch.load(ckpt_dir + f'/{args.source_data}_seed{args.seed}.ckpt')
     
    state_dict = checkpt['model'] 
    model_dict = model.state_dict()  
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict} 
    model_dict.update(state_dict)  
    model.load_state_dict(state_dict)
    model.to(device)
 
    model.eval()


    with torch.no_grad():
        emb = []
        yhat0_lis = []
        y0_lis = []
        obs = []
        for ex in range(num_train_batch):
            ys_obs = get_next_batch_new(train_ys_obs).to(device)
            ys = get_next_batch_new(train_ys).to(device) 
       
            z0, z0_std = model.encoder(ys_obs) 
            yhat0 = model.decoder(z0)
            #yhat0 = z0.mean(-1)
            emb.append(z0.detach().cpu().numpy())
            obs.append(ys_obs[:,:,0].detach().cpu().numpy())
            yhat0_lis.append(yhat0.detach().cpu().numpy())
            y0_lis.append(ys[:,0].detach().cpu().numpy())
        emb = np.concatenate(emb, axis=0)
        obs = np.concatenate(obs, axis=0)
        yhat0_lis = np.concatenate(yhat0_lis, axis=0)
        y0_lis = np.concatenate(y0_lis, axis=0)
    dist = []
    dist_hid = []
    for i in range(emb.shape[0]):
        for j in range(i,emb.shape[0]):
            dist.append( np.linalg.norm(obs[i]-obs[j]) )    
            dist_hid.append( np.linalg.norm(emb[i]-emb[j]) )
  
    fig, axs = plt.subplots(1,2,figsize=(6,3))
    axs[0].plot(dist, dist_hid, '.')
    axs[0].set_xlabel('L2 Distance in Input Space')
    axs[0].set_ylabel('L2 Distance in Hidden Space')
    axs[1].plot(y0_lis.reshape(-1), yhat0_lis.reshape(-1), '.')
    axs[1].set_xlabel('True IC')
    axs[1].set_ylabel('Predicted IC')
    fig.tight_layout()
    plt.savefig(f"{args.train_dir}/IC_{global_step}.png")
 
def hinge(z):
    return max(1-z,0)
    
def main():
    def plot(): 
        with torch.no_grad():
            for ex in range(num_test_batch):
                 
                ys_obs = get_next_batch_new(test_ys_obs).to(device)
                ys = get_next_batch_new(test_ys).to(device)
                                            # previous time list: ts_vis_ 
                zs = model.sample_q(ys_obs, ts=ts, batch_size=vis_batch_size, eps=eps ).squeeze()
                # [T, N_sample, vis_batch_size]
                #ys = shift_inverse(ys, ys_obs[:,-1,0])
                #zs = shift_inverse(zs.transpose(1,0), ys_obs[:,-1,0]).transpose(1,0)
                ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy() 
                ys_ = ys.cpu().numpy()
                ys_obs_ = ys_obs.cpu().numpy()
                # np.save(f"{args.train_dir}/plot/batch{ex}_ts_vis.npy", ts_vis_)
                # np.save(f"{args.train_dir}/plot/batch{ex}_zs.npy", zs_)
                # np.save(f"{args.train_dir}/plot/batch{ex}_ys.npy", ys_)
                # np.save(f"{args.train_dir}/plot/batch{ex}_ys_obs.npy", ys_obs_)
                for i in range( min(4,ys.shape[0])):
                    img_path = os.path.join(args.train_dir, f'plot/batch{ex}_sample{i}_global_step_{global_step}.png')
                    samples_ = zs_[:,i, vis_idx] 
                    zs_ = np.sort(zs_, axis=1)
                    #plt.subplot(frameon=True)
                    fig, axs = plt.subplots(1,1,figsize=(6, 5))
                    if args.show_percentiles:
                        for alpha, percentile in zip(alphas, percentiles):
                            idx = int((1 - percentile) / 2. * vis_batch_size)
                            zs_bot_, zs_top_ = zs_[:, i,idx], zs_[:, i,-idx]
                            # previous time list: ts_vis_ 
                            axs.fill_between(ts_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)
                    if args.show_mean:
                        # previous time list: ts_vis_ 
                        axs.plot(ts_, zs_[:,i,:].mean(axis=1), color=mean_color, label=f"{args.diff_eq}")
                    if args.show_samples:
                        for j in range(num_samples):
                            # previous time list: ts_vis_ 
                            axs.plot(ts_, samples_[:, j], color=sample_colors[j], linewidth=1.0)
                    if args.show_arrows:
                        num, dt = 12, 0.12
                        t, y = torch.meshgrid(
                            [torch.linspace(ts_vis_[0]+0.01, ts_vis_[-1]-0.01, num).to(device), torch.linspace(0, 1.5, num).to(device)]
                        )
                        t, y = t.reshape(-1, 1), y.reshape(-1, 1)
                        fty = model.f(t=t, y=y).reshape(num, num)
                        dt = torch.zeros(num, num).fill_(dt).to(device)
                        dy = fty * dt
                        dt_, dy_, t_, y_ = dt.cpu().numpy(), dy.cpu().numpy(), t.cpu().numpy(), y.cpu().numpy()
                        axs.quiver(t_, y_, dt_, dy_, alpha=0.3, edgecolors='k', width=0.0035, scale=50)
                    if args.hide_ticks:
                        axs.set_xticks([], [])
                        axs.set_yticks([], [])
                     
                    axs.plot(ts_, ys_[i], '-', zorder=3, color='k', label='Target' )   
                    axs.scatter(ts_vis_[:args.cond_len], ys_obs_[i,:,0], marker='o', zorder=3, color='k', label='Data', s=25)   
                    axs.axvline(ts_vis_[args.cond_len], color='#BDBDBD', linewidth=1)
                    axs.set_ylim(ylims)
                    axs.set_xlabel('$t$')
                    axs.set_ylabel('$Y_t$')
                    axs.text(0.1, 0.9, f"Best MAPE {best_mape}\nBest RMSE {best_rmse}\nTrain RT {train_rt}\nTest RT {test_rt}", fontsize=10, transform=axs.transAxes, va='top')
                    axs.legend()
                    fig.tight_layout()
                    plt.savefig(img_path, dpi=args.dpi)
                    plt.close()
                    print(f'Saved figure at: {img_path}')
                    #logging.info(f'Saved figure at: {img_path}')
                 
                 
    # Dataset.
    ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, train_ys, train_ys_obs, test_ys, test_ys_obs, num_train_batch, num_test_batch, min_, max_ = make_data()
     
    # Plotting parameters.
    vis_batch_size = 1024
    ylims = (min_-0.1, max_+0.1)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    vis_idx = np.random.permutation(vis_batch_size)
    # From https://colorbrewer2.org/.
    if args.color == "blue":
        sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
        fill_color = '#9ebcda'
        mean_color = '#1B5EFF'
        num_samples = len(sample_colors)
    else:
        sample_colors = ('#fc4e2a', '#e31a1c', '#bd0026')
        fill_color = '#fd8d3c'
        mean_color = '#800026'
        num_samples = len(sample_colors)
     
    eps = torch.randn(vis_batch_size, 1).to(device) # can be shared by all samples # Fix seed for the random draws used in the plots.
    bm = torchsde.BrownianInterval(
        t0=ts_vis[0],
        t1=ts_vis[-1],
        size=(vis_batch_size*args.train_batch_size, args.latent_dim),
        device=device,
        levy_area_approximation='space-time'
    )  # We need space-time Levy area to use the SRK solver
    
    # Model.
    if args.diff_eq == 'sde':
        args.method = 'euler'
        model = LatentSDE(mu=args.mean, sigma=args.std, dim_hid=args.latent_dim).to(device)
    else:
        args.method = 'rk4'
        model = LatentODE(mu=args.mean, sigma=args.std, dim_hid=args.latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    kl_scheduler = LinearScheduler(iters=args.kl_anneal_iters)

    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric() 
    best_mape = np.inf
    best_rmse = np.inf
    best_epo = 0 

    train_rt = 0
    test_rt = 0
    logpy_lis = []
    kl_lis = []
    loss_lis = []
    mape_lis = []
    rmse_lis = []
    test_mape = []
    test_rmse = []
    for global_step in tqdm.tqdm(range(args.train_iters)):
        if best_epo + 50 < global_step:
            logging.info('Early Stopping')
            break 
        # Plot and save. 
        # Train.
        mape_tot = 0
        rmse_tot = 0
        count = 0
        t1 = time.time()

        logpy_tot = 0
        kl_tot = 0
        loss_tot = 0
        mape_tot = 0
        rmse_tot = 0
        model.train()
        for ex in range(num_train_batch):
            ys_obs = get_next_batch_new(train_ys_obs).to(device)
            ys = get_next_batch_new(train_ys).to(device)
            count += ys_obs.shape[0]
            optimizer.zero_grad()
            zs, kl = model(ys_obs, ts=ts_ext, batch_size=args.batch_size)
            
            # plt.clf()
            # plt.plot(zs[0,:,0].detach().cpu().numpy(), ys[:,0,0].detach().cpu().numpy(), '.')
            # plt.show()
            #zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.
            ## zs: [T, N_sample, vis_batch_size]
            ## ys: [N_sample, T, 1]
            likelihood_constructor = {"laplace": distributions.Laplace, "normal": distributions.Normal}[args.likelihood]
            likelihood = likelihood_constructor(loc=zs, scale=args.scale)
            logpy = likelihood.log_prob(ys.transpose(1,0)).sum(dim=0).mean(dim=0).mean()
            #err_ic = torch.square(zs[0].mean(-1) - ys[:,0,0]).mean() # IC 

            # ranking_loss = 0
            # o = zs[0].mean(-1)
            # y =  ys[:,0,0]
            # for i in range(len(o)):
            #     for j in range(i,len(o)):
            #         ranking_loss += hinge((o[i] - o[j]) * torch.sign(y[i] - y[j]))

            #print(ranking_loss)
            loss =  -logpy + kl * kl_scheduler.val
            #loss = err_ic -logpy + kl * kl_scheduler.val
            # print('ranking_loss', ranking_loss)
            # print('-logpy', -logpy)
            # print('kl', kl)  

            loss.backward()

            optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            logpy_metric.step(logpy)
            kl_metric.step(kl)
            loss_metric.step(loss)

            true = ys.squeeze(-1).detach().cpu().numpy()
            pred = zs.detach().cpu().numpy()
            last_obs = ys_obs[:,-1,0].detach().cpu().numpy()
            pred = pred.mean(-1).T 
            #true = shift_inverse(true, last_obs)
            #pred = shift_inverse(pred, last_obs) 
            cur_mape = mape( true, pred)
            cur_rmse = rmse( true, pred)
            mape_tot += (cur_mape.sum())
            rmse_tot += (cur_rmse.sum()) 
            logpy_tot    += logpy_metric.val
            kl_tot   += kl_metric.val
            loss_tot += loss_metric.val 
        logging.info(
            f'global_step: {global_step}, '
            f'logpy: {logpy_tot/num_train_batch:.3f}, '
            f'kl: {kl_tot/num_train_batch:.3f}, '
            f'loss: {loss_tot/num_train_batch:.3f}, '
            f'MAPE {mape_tot/count}, '
            f'RMSE {rmse_tot/count}'
        )
 
        logpy_lis.append(logpy_tot/num_train_batch)
        kl_lis.append(kl_tot/num_train_batch)
        loss_lis.append(loss_tot/num_train_batch)
        mape_lis.append(mape_tot/count)
        rmse_lis.append(rmse_tot/count)

        t2 = time.time()
        train_rt += t2 - t1
        # if global_step == args.train_iters - 1:
        #     print('----plot----')
        #     plot() 
        # ####################### TEST
        model.eval()
        t1 = time.time()
        with torch.no_grad():
            mape_tot = 0
            rmse_tot = 0
            count = 0
            true_lis = []
            pred_lis = []
            for ex in range(num_test_batch):
                ys_obs = get_next_batch_new(test_ys_obs).to(device)
                ys = get_next_batch_new(test_ys).to(device)
                count += ys_obs.shape[0]
                zs, kl = model(ys_obs, ts=ts_ext, batch_size=args.batch_size)
                #zs = zs.squeeze()
                #zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.
                true = ys.squeeze(-1).detach().cpu().numpy()
                pred = zs.detach().cpu().numpy()
                last_obs = ys_obs[:,-1,0].detach().cpu().numpy()
                pred = pred.mean(-1).T 
                #true = shift_inverse(true, last_obs)
                #pred = shift_inverse(pred, last_obs)
                true_lis.append(true)
                pred_lis.append(pred)
                cur_mape = mape( true, pred)
                cur_rmse = rmse( true, pred)
                
                mape_tot += (cur_mape.sum())
                rmse_tot += (cur_rmse.sum())
            if mape_tot/count < best_mape:
                best_mape = mape_tot/count
                best_rmse = rmse_tot/count
                best_epo = global_step
                true_lis = np.concatenate(true_lis, axis=0)
                pred_lis = np.concatenate(pred_lis, axis=0)
                np.save(f"{args.train_dir}/true.npy", true_lis)
                np.save(f"{args.train_dir}/pred.npy", pred_lis)
                if args.save_ckpt:
                    torch.save(
                        {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'kl_scheduler': kl_scheduler},
                        os.path.join(ckpt_dir, f'{args.source_data}_seed{args.seed}.ckpt')
                    )
                 
            logging.info( 
                f'Test MAPE {mape_tot/count}, '
                f'Test RMSE {rmse_tot/count}\n'
                f'Best MAPE {best_mape}, '
                f'Best RMSE {best_rmse}\n'
            )

            test_mape.append(mape_tot/count)
            test_rmse.append(rmse_tot/count)
        t2 = time.time()
         
        test_rt += t2 - t1 

        # if global_step % args.pause_iters == 0 and global_step > 0:
        #     test(global_step, model) 
             
        # if global_step == args.train_iters - 1:
        #     test(global_step, model)  
    train_rt = train_rt / args.train_iters    
    test_rt = test_rt / args.train_iters
    logging.info( 
        f'Train runtime per epoch {train_rt}, '
        f'Test runtime {test_rt}\n' 
    )
    np.savetxt(f"{args.train_dir}/final_res.txt", [best_mape, best_rmse, train_rt, test_rt])
    np.savetxt(f"train_stats.txt", 
               np.stack((logpy_lis, kl_lis, loss_lis, mape_lis, rmse_lis)).T,
               header='likelihood,kl,loss,mape,rmse',delimiter=',',comments='')
    np.savetxt(f"test_stats.txt", 
               np.stack((test_mape,test_rmse)).T, 
               header='test_mape,test_rmse', delimiter=',',comments='')
    fig, axs = plt.subplots(1,5,figsize=(15,3))
    axs[0].plot(logpy_lis)
    axs[1].plot(kl_lis)
    axs[2].plot(loss_lis)
    axs[3].plot(mape_lis)
    axs[4].plot(rmse_lis)
    axs[0].set_title('Likelihood')
    axs[1].set_title('KL Divergence')
    axs[2].set_title('Loss')
    axs[3].set_title('MAPE')
    axs[4].set_title('RMSE')
    fig.tight_layout()
    for i in range(5):
        axs[i].set_xlabel('Epoch')
    plt.savefig(f"{args.train_dir}/train_stats.png")

    fig, axs = plt.subplots(1,2,figsize=(6,3))
    axs[0].plot(test_mape)
    axs[1].plot(test_rmse)
    axs[0].set_title('Test MAPE')
    axs[1].set_title('Test RMSE')
    for i in range(2):
        axs[i].set_xlabel('Epoch')
    fig.tight_layout()
    plt.savefig(f"{args.train_dir}/test_stats.png")

 
 
if __name__ == '__main__':
    # The argparse format supports both `--boolean-argument` and `--boolean-argument True`.
    # Trick from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse.
    parser = argparse.ArgumentParser()
    #parser.add_argument('--no-gpu', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--cuda_id', type=int, default=0 )
    parser.add_argument('--debug', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, default='lc-ode')
    parser.add_argument('--save-ckpt', type=str2bool, default=True, const=True, nargs="?")

    parser.add_argument('--data', type=str, default='cnn', choices=['mlp', 'cnn'])
    parser.add_argument('--kl-anneal-iters', type=int, default=100, help='Number of iterations for linear KL schedule.')
    parser.add_argument('--train-iters', type=int, default=150, help='Number of iterations for training.')
    parser.add_argument('--pause-iters', type=int, default=50, help='Number of iterations before pausing.')
    parser.add_argument('--batch-size', type=int, default=512, help='Number of sampled trajectories.')
    parser.add_argument('--train-batch-size', type=int, default=4, help='Number of sampled trajectories.')
    
    parser.add_argument('--likelihood', type=str, choices=['normal', 'laplace'], default='normal')
    parser.add_argument('--scale', type=float, default=0.05, help='Scale parameter of Normal and Laplace.')

    parser.add_argument('--adjoint', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--method', type=str, default='rk4', choices=('euler', 'milstein', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-3)

    parser.add_argument('--show-prior', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-samples', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-percentiles', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-arrows', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-mean', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--hide-ticks', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--color', type=str, default='blue', choices=('blue', 'red'))
    parser.add_argument('--cond_len', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--source_data', type=str, default='cifar10')
    parser.add_argument('--diff_eq', type=str, default='ode')
    parser.add_argument('--curve_type', type=str, default='loss')
      
    args = parser.parse_args()

    args.train_dir =  f"./wograph_data_fixed/{args.source_data}_{args.curve_type}_{args.diff_eq}_seed{args.seed}/{today}"

    ckpt_dir = os.path.join(args.train_dir, 'ckpts')
    plot_dir = os.path.join(args.train_dir, 'plot')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device('cpu')
    device = torch.device(f"cuda:{args.cuda_id}")
    #device = torch.device('cpu')
    print(device)
    manual_seed(args.seed)
    with open(f'{args.train_dir}/res.log', 'w') as f:
        # wipe out
        pass
    logging.basicConfig(filename=f'{args.train_dir}/res.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'  )

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

     
    odeint_fn = odeint_adjoint if args.adjoint else odeint
    sdeint_fn = torchsde.sdeint_adjoint if args.adjoint else torchsde.sdeint

    main() 

