import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sparse_modules import SparseLinear,SparseParameter,SparseLinearMulti_mask,SparseParameterMulti_mask
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from models.networks.dmgin_layer import GINLayer, ApplyNodeFunc, MLP
import pdb

def resetBN_is_score(tmpBN):
    tmpBN.weight.is_score = True
    tmpBN.bias.is_score = True
    tmpBN.weight.sparsity = 0.0
    tmpBN.bias.sparsity = 0.0
    return tmpBN


def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()
class dmGINNet(nn.Module):
    
    def __init__(self, args, graph):
        super().__init__()
        #in_dim = net_params[0]
        #hidden_dim = net_params[1]
        #n_classes = net_params[2]
        self.args=args
        in_dim=args.num_feats
        hidden_dim=args.dim_hidden
        n_classes=args.num_classes
        #dropout = 0.5
        dropout=args.dropout
        #self.n_layers = 2
        self.n_layers=args.num_layers
        
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1               # GIN
        learn_eps = False              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False      
        batch_norm = False
        residual = False
        self.n_classes = n_classes
        self.add_loop = False
        self.eps = torch.nn.Parameter(torch.empty(1))
        #self.eps.is_score = True
        #self.eps.sparsity = 0.0
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.ginlayersBN = torch.nn.ModuleList()
        self.BN_GIN = args.BN_GIN
        self.train_mode = args.train_mode

        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim,args=args)
                if args.BN_GIN and args.train_mode == 'score_only':
                    bn_node_h = resetBN_is_score(nn.BatchNorm1d(hidden_dim))
                elif args.BN_GIN and args.train_mode != 'score_only':
                    bn_node_h = nn.BatchNorm1d(hidden_dim)
                else:
                    bn_node_h = None
            elif layer<(self.n_layers-1):
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim,args=args)
                if args.BN_GIN and args.train_mode == 'score_only':
                    bn_node_h = resetBN_is_score(nn.BatchNorm1d(hidden_dim))
                elif args.BN_GIN and args.train_mode != 'score_only':
                    bn_node_h = nn.BatchNorm1d(hidden_dim)
                else:
                    bn_node_h = None
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes,args=args)
                if args.BN_GIN and args.train_mode == 'score_only':
                    bn_node_h = resetBN_is_score(nn.BatchNorm1d(n_classes))
                elif args.BN_GIN and args.train_mode != 'score_only':
                    bn_node_h = nn.BatchNorm1d(n_classes)
                else:
                    bn_node_h = None
                
            self.ginlayers.append(mlp)
            self.ginlayersBN.append(bn_node_h)

    def get_threshold(self,sparsity):
        if(self.args.enable_mask == True): # enable multi-mask
            threshold_list = []
            for value in sparsity:
                local=[]
                for name, p in self.named_parameters():
                    if hasattr(p, 'is_score') and p.is_score:
                        local.append(p.detach().flatten())
                local=torch.cat(local)
                if self.args.enable_abs_comp== False:
                    threshold=percentile(local,value*100)
                else:
                    threshold=percentile(local.abs(),value*100)
                threshold_list.append(threshold)
            return threshold_list
        else:
            local=[]
            for name, p in self.named_parameters():
                if hasattr(p, 'is_score') and p.is_score:
                    local.append(p.detach().flatten())
            local=torch.cat(local)
            if self.args.enable_abs_comp== False:
                threshold=percentile(local,sparsity*100)
            else:
                threshold=percentile(local.abs(),sparsity*100)
            return threshold     
        
    def forward(self, g, h, snorm_n, snorm_e,sparsity=None):
        if sparsity is None:
            if (self.args.enable_mask == True):
                sparsity = self.args.sparsity_list
            else:
                sparsity=self.args.linear_sparsity
        threshold=self.get_threshold(sparsity)     
        adj = g.adjacency_matrix
        h = h.unsqueeze(0) if h.dim() == 2 else h
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()
        for i in range(self.n_layers):
            out = torch.matmul(adj, h) # AX
            if self.add_loop:
                out = (1 + self.eps) * h + out  # (A+(1+eps)*I)*X
            h = self.ginlayers[i](out,threshold=threshold) # (AX) W

            #if self.BN_GIN: # BN 
            #    h = h.squeeze(0)
            #    h = self.ginlayersBN[i](h)
            #    h = h.unsqueeze(0)
            #if (i != (self.n_layers-1)):
            #    h = F.relu(h) # Relu
        h = h.squeeze(0)
        return h
      