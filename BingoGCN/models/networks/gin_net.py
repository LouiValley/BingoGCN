# import dgl
import math

import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F
# from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from .sparse_modules import (
    NMSparseLinear,
    NMSparseMultiLinear,
    SparseLinear,
    SparseLinearMulti_mask,
)

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019) # noqa
    https://arxiv.org/pdf/1810.00826.pdf
"""

# import pdb

from .gin_layer import MLP, ApplyNodeFunc, GINLayer


def percentile(t, q):
    t_flat = t.view(-1)
    t_sorted, _ = torch.sort(t_flat)
    k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
    return t_sorted[k].item()


def calculate_sparsity(x):
    # GPU上のテンソルをCPUに移動し、勾配情報を切り離した後、NumPy配列に変換
    x_cpu = x.detach().cpu().numpy()  # xがGPU上にある場合に必要
    # 0の数をカウント
    zero_count = np.count_nonzero(x_cpu == 0)
    # 全要素数
    total_elements = x_cpu.size
    # sparsityの計算
    sparsity_ratio = zero_count / total_elements
    return sparsity_ratio


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        # k_val = percentile(scores, sparsity*100)
        # if glob:
        out = torch.where(
            scores < threshold, zeros.to(scores.device), ones.to(scores.device)
        )
        # else:
        #    k_val = percentile(scores, threshold*100)
        #    out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class GINNet(nn.Module):

    def __init__(self, args, graph, n_parts=None):
        super().__init__()
        # in_dim = net_params[0]
        # hidden_dim = net_params[1]
        # n_classes = net_params[2]
        self.args = args
        in_dim = args.num_feats
        hidden_dim = args.dim_hidden
        n_classes = args.num_classes
        # dropout = 0.5
        dropout = args.dropout
        # self.n_layers = 2
        self.n_layers = args.num_layers

        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1  # GIN
        learn_eps = False  # GIN
        neighbor_aggr_type = "mean"  # GIN
        graph_norm = False
        batch_norm = False
        residual = False
        self.n_classes = n_classes

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.n_parts = n_parts

        if self.args.gra_part:
            self.hidden_reps = [[] for _ in range(n_parts)]

        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(
                    n_mlp_layers,
                    in_dim,
                    hidden_dim,
                    hidden_dim,
                    args=args,
                    layer=layer,
                )
            elif layer < (self.n_layers - 1):
                mlp = MLP(
                    n_mlp_layers,
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    args=args,
                    layer=layer,
                )
            else:
                mlp = MLP(
                    n_mlp_layers,
                    hidden_dim,
                    hidden_dim,
                    n_classes,
                    args=args,
                    layer=layer,
                )

            self.ginlayers.append(
                GINLayer(
                    ApplyNodeFunc(mlp),
                    neighbor_aggr_type,
                    dropout,
                    graph_norm,
                    batch_norm,
                    residual,
                    0,
                    learn_eps,
                    args,
                    layer=layer,
                )
            )

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score
        if (
            not args.unstructured_for_last_layer
            and args.nmsparsity
            and args.enable_mask
        ):
            self.linears_prediction = NMSparseMultiLinear(
                hidden_dim, n_classes, args=args, layer=layer + 1
            )
        elif not args.unstructured_for_last_layer and args.nmsparsity:
            self.linears_prediction = NMSparseLinear(
                hidden_dim, n_classes, args=args, layer=layer + 1
            )
        elif args.enable_mask is True:
            self.linears_prediction = SparseLinearMulti_mask(
                hidden_dim, n_classes, args=args, layer=layer + 1
            )
        else:
            self.linears_prediction = SparseLinear(
                hidden_dim, n_classes, args=args, layer=layer + 1
            )

        # if args.enable_mask is True:
        #     self.linears_prediction = SparseLinearMulti_mask(
        #         hidden_dim, n_classes, bias=False, args=args, layer=layer + 1
        #     )
        # else:
        #     self.linears_prediction = SparseLinear(
        #         hidden_dim, n_classes, bias=False, args=args, layer=layer + 1
        #     )
        # self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        self.adj_mask2_fixed = nn.Parameter(
            torch.ones(self.edge_num, 1), requires_grad=False
        )

        if self.args.enable_feat_pruning:
            if self.args.x_pruning_layer == 0:
                self.x_weight = nn.Parameter(torch.ones(1, self.num_feats))
            else:
                self.x_weight = nn.Parameter(
                    torch.ones(1, self.args.dim_hidden)
                )

            self.init_param_(
                self.x_weight,
                init_mode=self.args.init_mode_mask,
                scale=self.args.init_scale_score,
            )
            self.x_weight_zeros = torch.zeros(self.x_weight.size())
            self.x_weight_ones = torch.ones(self.x_weight.size())
            self.x_weight.is_score = True
            self.x_weight.sparsity = self.args.featsparsity_ratio
            self.x_weight.requires_grad = True
            self.x_weight_zeros.requires_grad = False
            self.x_weight_ones.requires_grad = False

        if self.args.enable_node_pruning:
            self.x_weight = nn.Parameter(torch.ones(self.args.num_nodes, 1))
            self.init_param_(
                self.x_weight,
                init_mode=self.args.init_mode_mask,
                scale=self.args.init_scale_score,
            )
            self.x_weight_zeros = torch.zeros(self.x_weight.size())
            self.x_weight_ones = torch.ones(self.x_weight.size())
            self.x_weight.is_score = True
            self.x_weight.sparsity = self.args.featsparsity_ratio
            self.x_weight.requires_grad = True
            self.x_weight_zeros.requires_grad = False
            self.x_weight_ones.requires_grad = False

    def forward(
        self,
        g,
        h,
        snorm_n,
        snorm_e,
        sparsity=None,
        epoch=None,
        return_intermediate=None,
        layer_idx=None,
        edge_weight=None,
        part_idx=None,
    ):
        if sparsity is None:
            if self.args.enable_mask is True:
                sparsity = self.args.sparsity_list
            else:
                sparsity = self.args.linear_sparsity
        threshold = self.get_threshold(sparsity)

        if self.args.x_pruning_layer == 0 and (
            self.args.enable_feat_pruning or self.args.enable_node_pruning
        ):
            if self.args.enable_mask is True:
                x_pruning_ratio = self.args.featsparsity_ratio * (
                    sparsity[0] / self.args.linear_sparsity
                )
            else:
                x_pruning_ratio = self.args.featsparsity_ratio * (
                    sparsity / self.args.linear_sparsity
                )

            # x_maskを適用する前のsparsityを計算
            before_sparsity = calculate_sparsity(h)
            print(f"before: {before_sparsity:.8f}")

            x_mask = self.get_feat_mask(
                x_pruning_ratio, self.x_weight, epoch=epoch
            )
            h = h * x_mask

            # x_maskを適用した後のsparsityを計算
            after_sparsity = calculate_sparsity(h)
            print(f"after: {after_sparsity:.8f}")

        if self.args.gra_part is True:
            self.edge_num = g.number_of_edges()
            adj_mask2_fixed_new = nn.Parameter(
                torch.ones(self.edge_num, 1), requires_grad=False
            ).to(g.device)
            g.edata["mask"] = adj_mask2_fixed_new
            # adjacency_matrix_list = g.adjacency_matrix.tolist()
            # with open("g_sub_adjacency_matrix.txt", "w") as file:
            #     for row in adjacency_matrix_list:
            #         row_str = " ".join(str(val) for val in row)
            #         file.write(row_str + "\n")
        else:
            g.edata["mask"] = self.adj_mask2_fixed
            # output_list = self.adj_mask2_fixed.tolist()
            # with open(f"self.adj_mask2_fixed.txt", "w") as file:
            #     for item in output_list:
            #         file.write("%s\n" % item)
            # adjacency_matrix_list = g.adjacency_matrix.tolist()
            # with open(f"g_adjacency_matrix.txt", "w") as file:
            #     for row in adjacency_matrix_list:
            #         row_str = " ".join(str(val) for val in row)
            #         file.write(row_str + "\n")

        hidden_rep = []

        for i in range(self.n_layers):
            if return_intermediate:
                if i == layer_idx:
                    h = self.ginlayers[i](
                        g, h, snorm_n, threshold=threshold, sparsity=sparsity
                    )
                    self.hidden_reps[part_idx].append(h)
                    if self.args.x_pruning_layer - 1 == i and (
                        self.args.enable_feat_pruning
                        or self.args.enable_node_pruning
                    ):
                        if self.args.enable_mask is True:
                            x_pruning_ratio = self.args.featsparsity_ratio * (
                                sparsity[0] / self.args.linear_sparsity
                            )
                        else:
                            x_pruning_ratio = self.args.featsparsity_ratio * (
                                sparsity / self.args.linear_sparsity
                            )

                        # x_maskを適用する前のsparsityを計算
                        before_sparsity = calculate_sparsity(h)
                        print(f"before: {before_sparsity:.8f}")

                        x_mask = self.get_feat_mask(
                            x_pruning_ratio, self.x_weight, epoch=epoch
                        )
                        h = h * x_mask

                        # x_maskを適用した後のsparsityを計算
                        after_sparsity = calculate_sparsity(h)
                        print(f"after: {after_sparsity:.8f}")
                    if i + 1 != self.args.num_layers:
                        return h

                else:
                    pass

            else:
                h = self.ginlayers[i](
                    g, h, snorm_n, threshold=threshold, sparsity=sparsity
                )
                hidden_rep.append(h)
                if self.args.x_pruning_layer - 1 == i and (
                    self.args.enable_feat_pruning
                    or self.args.enable_node_pruning
                ):
                    if self.args.enable_mask is True:
                        x_pruning_ratio = self.args.featsparsity_ratio * (
                            sparsity[0] / self.args.linear_sparsity
                        )
                    else:
                        x_pruning_ratio = self.args.featsparsity_ratio * (
                            sparsity / self.args.linear_sparsity
                        )

                    # x_maskを適用する前のsparsityを計算
                    before_sparsity = calculate_sparsity(h)
                    print(f"before: {before_sparsity:.8f}")

                    x_mask = self.get_feat_mask(
                        x_pruning_ratio, self.x_weight, epoch=epoch
                    )
                    h = h * x_mask

                    # x_maskを適用した後のsparsityを計算
                    after_sparsity = calculate_sparsity(h)
                    print(f"after: {after_sparsity:.8f}")
                # output_list = h.tolist()
                # with open(f"h_layer{i}.txt", "w") as file:
                #     for item in output_list:
                #         file.write("%s\n" % item)

        if return_intermediate:
            if i + 1 == self.args.num_layers:
                score_over_layer = (
                    self.linears_prediction(
                        self.hidden_reps[part_idx][-2], threshold=threshold
                    )
                    + self.hidden_reps[part_idx][-1]
                ) / 2
                return score_over_layer
            else:
                pass
        else:
            # score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
            score_over_layer = (
                self.linears_prediction(
                    hidden_rep[-2], threshold=threshold, sparsity=sparsity
                )
                + hidden_rep[-1]
            ) / 2

        # output_list = score_over_layer.tolist()
        # with open(f"score_over_layer{layer_idx}.txt", "w") as file:
        #     for item in output_list:
        #         file.write("%s\n" % item)
        return score_over_layer

    def reset_hidden_reps(self):
        self.hidden_reps = [[] for _ in range(self.n_parts)]

    def init_param_(
        self, param, init_mode=None, scale=None, sparse_value=None
    ):
        if init_mode == "kaiming_normal":
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            param.data *= scale
        elif init_mode == "uniform":
            nn.init.uniform_(param, a=-1, b=1)
            param.data *= scale
        elif init_mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            param.data *= scale
        elif init_mode == "kaiming_normal_SF":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            param.data.normal_(0, std)
        elif init_mode == "signed_constant":
            # From github.com/allenai/hidden-networks
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale
        elif (
            init_mode == "signed_constant_sparse"
            or init_mode == "signed_constant_SF"
        ):
            # From okoshi'san's M-sup paper
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale  # scale value is defined in defualt as 1.0
        else:
            raise NotImplementedError

    def get_threshold(self, sparsity, epoch=None):
        if self.args.enable_mask is True:  # enable multi-mask
            threshold_list = []
            for value in sparsity:
                local = []
                for name, p in self.named_parameters():
                    if (
                        hasattr(p, "is_score")
                        and p.is_score
                        and p.sparsity == self.args.linear_sparsity
                    ):
                        local.append(p.detach().flatten())
                        # print('---para in calculating scores---')
                        # print(name)
                local = torch.cat(local)
                # threshold=percentile(local,sparsity*100)
                if self.args.enable_abs_comp is False:
                    threshold = percentile(local, value * 100)
                else:
                    threshold = percentile(local.abs(), value * 100)
                threshold_list.append(threshold)
            return threshold_list
        else:
            local = []
            for name, p in self.named_parameters():
                if (
                    hasattr(p, "is_score")
                    and p.is_score
                    and p.sparsity == self.args.linear_sparsity
                ):
                    local.append(p.detach().flatten())
                    # print('---para in calculating scores---')
                    # print(name)
            local = torch.cat(local)
            if self.args.enable_abs_comp is False:
                threshold = percentile(local, sparsity * 100)
            else:
                threshold = percentile(local.abs(), sparsity * 100)
            return threshold
        # if epoch!=None and (epoch+1)%50==0:
        #     print("sparsity",sparsity,"threshold",threshold)
        #     total_n=0.0
        #     total_re=0.0
        #     for name, p in self.named_parameters():
        #         if hasattr(p, 'is_score') and p.is_score and p.sparsity==self.args.linear_sparsity: # noqa
        #             mask=p.detach()<threshold
        #             mask=mask.float()
        #             total_re+=mask.sum().item()
        #             total_n+=mask.numel()
        #             print(name,":masked ratio",mask.sum().item()/mask.numel())
        #     print("total remove",total_re/total_n)

    def get_feat_mask(self, sparsity, x_weight, epoch=None):
        # local = []
        # local.append(x_weight.detach().flatten())
        # local = torch.cat(local)
        # threshold = torch.quantile(x_weight.abs(), sparsity)

        local = []
        local.append(x_weight.detach().flatten())
        local = torch.cat(local)
        threshold = percentile(local.abs(), sparsity * 100)
        mask = GetSubnet.apply(
            x_weight.abs(), threshold, self.x_weight_zeros, self.x_weight_ones
        )
        # mask = torch.where(
        #     x_weight.abs() < threshold,
        #     self.x_weight_zeros.to(x_weight.device),
        #     self.x_weight_ones.to(x_weight.device),
        # )

        return mask


"""
class GINNet_ss(nn.Module):

    def __init__(self, net_params, num_par):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        n_mlp_layers = 1               # GIN
        learn_eps = True              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False
        batch_norm = False
        residual = False
        self.n_classes = n_classes

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)


            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        self.classifier_ss = nn.Linear(hidden_dim, num_par, bias=False)

    def forward(self, g, h, snorm_n, snorm_e):

        # list of hidden representation at each layer (including input)
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        h_ss = self.classifier_ss(hidden_rep[0])

        return score_over_layer, h_ss

"""
