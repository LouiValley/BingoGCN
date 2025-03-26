import torch
import torch.nn.functional as F
from torch import nn
#from torch_geometric.nn import GCNConv
from models.networks.sparse_modules_graph import GCNConv
from models.networks.sparse_modules import SparseLinear,SparseParameter,SparseLinearMulti_mask,SparseParameterMulti_mask

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()

def resetBN_is_score(tmpBN):
    tmpBN.weight.is_score = True
    tmpBN.bias.is_score = True
    tmpBN.weight.sparsity = 0.0
    tmpBN.bias.sparsity = 0.0
    return tmpBN

class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class dmGCN(nn.Module):
    def __init__(self, args):
        super(dmGCN, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.args=args

         # first layer
        if (args.enable_mask == True):
            self.lin=SparseLinearMulti_mask(self.num_feats,self.dim_hidden,args=args)
        else:
            self.lin=SparseLinear(self.num_feats,self.dim_hidden,args=args)
        self.layers_GCN.append(self.lin)
        
        if self.type_norm == 'batch':
            if args.train_mode=='score_only':
                self.layers_bn.append(resetBN_is_score(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine)))
            else:
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())


        # middle layer
        for _ in range(self.num_layers - 2):
            if (args.enable_mask == True):
                self.lin=SparseLinearMulti_mask(self.dim_hidden,self.dim_hidden,args=args)
            else:
                self.lin=SparseLinear(self.dim_hidden,self.dim_hidden,args=args)
            self.layers_GCN.append(self.lin)

            if self.type_norm == 'batch':
                if args.train_mode=='score_only':
                    self.layers_bn.append(resetBN_is_score(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine)))
                else:
                    self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))
            elif self.type_norm == 'pair':
                self.layers_bn.append(pair_norm())

        # final layer
        if (args.enable_mask == True):
            self.lin=SparseLinearMulti_mask(self.dim_hidden,self.num_classes,args=args)
        else:
            self.lin=SparseLinear(self.dim_hidden,self.num_classes,args=args)
        self.layers_GCN.append(self.lin)

        if self.type_norm == 'batch':
            if args.train_mode=='score_only':
                self.layers_bn.append(resetBN_is_score(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine)))
            else:
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())

    def get_threshold(self,sparsity,epoch=None):
        if(self.args.enable_mask == True): # enable multi-mask
            threshold_list = []
            for value in sparsity:
                local=[]
                for name, p in self.named_parameters():
                    if hasattr(p, 'is_score') and p.is_score and p.sparsity==self.args.linear_sparsity:
                        local.append(p.detach().flatten())
                        #print('---para in calculating scores---')
                        #print(name)
                local=torch.cat(local)
                #threshold=percentile(local,sparsity*100)
                if self.args.enable_abs_comp== False:
                    threshold=percentile(local,value*100)
                else:
                    threshold=percentile(local.abs(),value*100)
                threshold_list.append(threshold)
            return threshold_list  
        else: 
            local=[]
            for name, p in self.named_parameters():
                if hasattr(p, 'is_score') and p.is_score and p.sparsity==self.args.linear_sparsity:
                    local.append(p.detach().flatten())
                    #print('---para in calculating scores---')
                    #print(name)
            local=torch.cat(local)
            if self.args.enable_abs_comp== False:
                threshold=percentile(local,sparsity*100)
            else:
                threshold=percentile(local.abs(),sparsity*100)
            return threshold  

    def forward(self, x, adj, sparsity=None,epoch=None):
        add_loop = True
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1  
            #adj[:, idx, idx] = 1 if not self.improved else 2

        if sparsity is None:
            if (self.args.enable_mask == True):
                sparsity = self.args.sparsity_list
            else:
                sparsity=self.args.linear_sparsity
        #if self.args.train_mode=='score_only':
        threshold=self.get_threshold(sparsity,epoch=epoch)
        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        adj = adj.squeeze(0)

        for i in range(self.num_layers - 1):
            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, threshold)   # out = self.lin(x): WX
            x = x.squeeze(0)
            x = torch.sparse.mm(adj, x)  #  out = A (WX)
            #x = torch.matmul(adj, x)  #  out = A (WX)

            # ....
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)

        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, threshold)  # out = self.lin(x): WX
        x = torch.sparse.mm(adj, x)  #  out = A (WX)
        return x
    def rerandomize(self,mode,la,mu):
        for m in self.modules():
            if type(m) is GCNConv:
                m.rerandomize(mode,la,mu)

