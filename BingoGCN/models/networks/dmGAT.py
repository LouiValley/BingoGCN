import torch
import torch.nn.functional as F
#from torch_geometric.nn import GATConv
from models.networks.sparse_modules_graph import GATConv, dmGATConv
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

class dmGAT(torch.nn.Module):
    def __init__(self, args):
        super(dmGAT, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        self.layers_bn = torch.nn.ModuleList([])
        self.args=args
        # space limit
        self.layers_GCN.append(dmGATConv(self.num_feats, self.dim_hidden,
                                       bias=False,concat=True,heads=args.heads,dropout=args.dropout,args=args))
        if self.type_norm == 'batch':
            if args.train_mode=='score_only':
                self.layers_bn.append(resetBN_is_score(torch.nn.BatchNorm1d(self.dim_hidden * args.heads,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine)))
            else:
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden * args.heads,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                dmGATConv(self.dim_hidden*args.heads, self.dim_hidden, bias=False, concat=True,heads=args.heads,dropout=args.dropout,args=args))
            if self.type_norm == 'batch':
                if args.train_mode=='score_only':
                    self.layers_bn.append(resetBN_is_score(torch.nn.BatchNorm1d(self.dim_hidden * args.heads,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine)))
                else:
                    self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden * args.heads,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))

        self.layers_GCN.append(dmGATConv(self.dim_hidden*args.heads, self.num_classes, bias=False,concat=False,dropout=args.dropout,heads=args.heads,args=args))
    
    def get_threshold(self,sparsity):
        if(self.args.enable_mask == True): # enable multi-mask
            threshold_list = []
            for value in sparsity:
                local = []
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
            local = []
            for name, p in self.named_parameters():
                if hasattr(p, 'is_score') and p.is_score:
                    local.append(p.detach().flatten())
            local=torch.cat(local)
            if self.args.enable_abs_comp== False:
                threshold=percentile(local,sparsity*100)
            else:
                threshold=percentile(local.abs(),sparsity*100)
            return threshold
        """
        print("sparsity",sparsity,"threshold",threshold)
        total_n=0.0
        total_re=0.0
        for name, p in self.named_parameters():
            if hasattr(p, 'is_score') and p.is_score:
                mask=p.detach()<threshold
                mask=mask.float()
                total_re+=mask.sum().item()
                total_n+=mask.numel()
                print(name,":masked ratio",mask.sum().item()/mask.numel())
        print("total remove",total_re/total_n)
        """

    def forward(self, x, adj,sparsity=None,epoch=0):
        if sparsity is None:
            if (self.args.enable_mask == True):
                sparsity = self.args.sparsity_list
            else:
                sparsity=self.args.linear_sparsity
        threshold=self.get_threshold(sparsity)
        
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, adj,threshold=threshold)
            if self.type_norm == 'batch':
                x = self.layers_bn[i](x)
            x = F.relu(x)

        x = self.layers_GCN[-1](x, adj,threshold)
        return x
    def rerandomize(self,mode,la,mu,sparsity=None):
        for m in self.modules():
            if type(m) is GATConv:
                m.rerandomize(mode,la,mu,sparsity)
