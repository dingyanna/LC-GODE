import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax,add_remaining_self_loops
import math 
from torch_geometric.nn import GCNConv, GATConv

class GTransLayer(MessagePassing):
    '''
    wrap up multiple layers
    '''
    def __init__(self, d_input=6, d_output=6, n_heads=1, dropout = 0.1,**kwargs):
        super(GTransLayer, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_output//n_heads
        self.d_q = d_output//n_heads
        self.d_e = d_output//n_heads
        self.d_sqrt = math.sqrt(d_output//n_heads)


        #Attention Layer Initialization
        self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for _ in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for _ in range(self.n_heads)])
        self.w_v_list = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for _ in range(self.n_heads)])
        self.res_block = nn.Linear(self.d_input, self.d_e)

        #Temporal Layer
        #self.temporal_net = TemporalEncoding(d_input)

        #Normalization
        self.layer_norm = nn.LayerNorm(d_input,elementwise_affine = False)

    def forward(self, x, edge_index, edge_attr, degree=None):
        ''' 
        :param x: [num_nodes, num_node_feat]
        :param edge_index: [2, num_edges]
        :param edge_attr: [num_edges, num_edge_feat] 
        :return:
        ''' 
        residual = x
        x = self.layer_norm(x)

        # Edge normalization if using multiplication
        #edge_weight = normalize_graph_asymmetric(edge_index,edge_weight,time_nodes.shape[0])
        #assert (torch.sum(edge_weight<0)==0) and (torch.sum(edge_weight>1) == 0)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edge_attr):
        ''' 
           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edge_attr: [num_edge,d] 
           :return:
        '''
        #print('\n\n>> x_i, x_j', x_i.shape, x_j.shape)
        #print('\n\n>>edge_index_i', edge_index_i.shape)
        #print('>>edge_attr', edge_attr.shape)
        
        messages = []
        for i in range(self.n_heads): 
            k_linear = self.w_k_list[i]
            q_linear = self.w_q_list[i]
            v_linear = self.w_v_list[i] 
  
            x_j_transfer = x_j  

            attention = self.each_head_attention(x_j_transfer,k_linear,q_linear,x_i) #[N_edge,1]
            attention = torch.div(attention,self.d_sqrt)
            #print('\n\n==> attention', attention.shape, '\n\n')

            # Need to multiply by original edge weight
            attention = attention * edge_attr    
            attention_norm = softmax(attention,edge_index_i) #[N_neighbors_,1]   
            sender = v_linear(x_j_transfer)

            message  = attention_norm * sender #[N_nodes,d]
            messages.append(message) 
        message_all_head  = torch.cat(messages,1) #[N_nodes, k*d] ,assuming K head
        #print('\nmessage_all_head', message_all_head.shape)
        return message_all_head

    def each_head_attention(self,x_j_transfer,w_k,w_q,x_i):
        '''

        :param x_j_transfer: sender [N_edge,d]
        :param w_k:
        :param w_q:
        :param x_i: receiver
        :return:
        ''' 
        # Receiver #[num_edge,d*heads]
        x_i = w_q(x_i)
        # Sender
        sender = w_k(x_j_transfer)
        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender,1),torch.unsqueeze(x_i,2)) #[N,1]

        return torch.squeeze( attention,1 )

    def update(self, aggr_out,residual): 
        # print(aggr_out.shape)
        # print(residual.shape)
        x_new = F.gelu(aggr_out) - self.res_block(residual)
        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__) 
  

class DiffPoolLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_clusters=1):
        super(DiffPoolLayer, self).__init__()

        
        self.gcn_embed = GCNConv(in_channels, out_channels)
        self.gcn_pool = GCNConv(in_channels, num_clusters)

    def forward(self, x, batch):
        # Node embeddings
        embed = torch.relu(self.gcn_embed(x, batch.edge_index, batch.edge_weight))

        # Compute assignment matrix
        score = self.gcn_pool(x, batch.edge_index, batch.edge_weight)
        
        n_graph = batch.topo.shape[0] 
         
        res = [ ]
        for i in range( n_graph):
            s = torch.softmax(score[ torch.sum(batch.topo[:i,0]): torch.sum(batch.topo[:i+1,0])  ], dim=0) # [N , 1]?
            global_emb = (s.reshape(1,-1) @ embed[ torch.sum(batch.topo[:i,0]): torch.sum(batch.topo[:i+1,0]) ]).reshape(-1)
            res.append(global_emb)
             
        res = torch.stack(res) # N x d 
         
        return res

class GraphEncoder(torch.nn.Module):
    def __init__(self, num_features, out_channel, args):
        super(GraphEncoder, self).__init__()
        #self.diffpool1 = DiffPoolLayer(num_features, out_channel//2, 5)
        if args.gnn == 'gcn':
            self.gcn1 = GCNConv(num_features, out_channel//2)
            self.gcn2 = GCNConv(out_channel//2, out_channel )
        elif args.gnn == 'gat':
            self.gcn1 = GATConv(num_features, out_channel//2)
            self.gcn2 = GATConv(out_channel//2, out_channel )
        else:
            self.gcn1 = GTransLayer(num_features, out_channel//2)
            self.gcn2 = GTransLayer(out_channel//2, out_channel)
        if args.graph_pooling == 'diff':
            self.pool_func = DiffPoolLayer(out_channel, out_channel , 1)
            self.forward = self.diffpool
        elif args.graph_pooling == 'mean':
            self.pool_func = global_mean_pool
            self.forward = self.pool
        else:
            self.pool_func = global_max_pool
            self.forward = self.pool
         
        self.final_gcn = GCNConv(out_channel, out_channel)
  
    def pool(self, batch):
        x = self.gcn1(batch.x, batch.edge_index, batch.edge_attr)
        x = self.gcn2(x, batch.edge_index, batch.edge_attr)
        x = self.pool_func(x, batch.batch)   
        return x
    
    def diffpool(self, batch):
        x = self.gcn1(batch.x, batch.edge_index, batch.edge_attr )
        x = self.gcn2(x, batch.edge_index, batch.edge_attr) 
        x = self.pool_func(x, batch) 
        return x
 