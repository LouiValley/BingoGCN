import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from torch_geometric.nn import GATConv
from .sparse_modules_graph import GATConv


def percentile(t, q):
    t_flat = t.view(-1)
    t_sorted, _ = torch.sort(t_flat)
    k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
    return t_sorted[k].item()


def resetBN_is_score(tmpBN):
    tmpBN.weight.is_score = True
    tmpBN.bias.is_score = True
    tmpBN.weight.sparsity = 0.0
    tmpBN.bias.sparsity = 0.0
    return tmpBN


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


class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        self.layers_bn = torch.nn.ModuleList([])
        self.args = args
        self.layers_GCN.append(
            GATConv(
                self.num_feats,
                self.dim_hidden,
                bias=False,
                concat=True,
                heads=args.heads,
                dropout=args.dropout,
                args=args,
                layer=0,
            )
        )
        if self.type_norm == "batch":
            if args.train_mode == "score_only":
                self.layers_bn.append(
                    resetBN_is_score(
                        torch.nn.BatchNorm1d(
                            self.dim_hidden,
                            momentum=args.bn_momentum,
                            track_running_stats=args.bn_track_running_stats,
                            affine=args.bn_affine,
                        )
                    )
                )
            else:
                self.layers_bn.append(
                    torch.nn.BatchNorm1d(
                        self.dim_hidden,
                        momentum=args.bn_momentum,
                        track_running_stats=args.bn_track_running_stats,
                        affine=args.bn_affine,
                    )
                )

        for i in range(self.num_layers - 2):
            self.layers_GCN.append(
                GATConv(
                    self.dim_hidden * args.heads,
                    self.dim_hidden,
                    bias=False,
                    concat=True,
                    heads=args.heads,
                    dropout=args.dropout,
                    args=args,
                    layer=i + 1,
                )
            )
            if self.type_norm == "batch":
                if args.train_mode == "score_only":
                    self.layers_bn.append(
                        resetBN_is_score(
                            torch.nn.BatchNorm1d(
                                self.dim_hidden,
                                momentum=args.bn_momentum,
                                track_running_stats=args.bn_track_running_stats,
                                affine=args.bn_affine,
                            )
                        )
                    )
                else:
                    self.layers_bn.append(
                        torch.nn.BatchNorm1d(
                            self.dim_hidden,
                            momentum=args.bn_momentum,
                            track_running_stats=args.bn_track_running_stats,
                            affine=args.bn_affine,
                        )
                    )

        self.layers_GCN.append(
            GATConv(
                self.dim_hidden * args.heads,
                self.num_classes,
                bias=False,
                concat=False,
                dropout=args.dropout,
                heads=args.heads,
                args=args,
                layer=self.num_layers - 1,
            )
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

    def get_threshold(self, sparsity):
        if self.args.enable_mask is True:  # enable multi-mask
            threshold_list = []
            for value in sparsity:
                local = []
                for name, p in self.named_parameters():
                    if hasattr(p, "is_score") and p.is_score:
                        local.append(p.detach().flatten())
                local = torch.cat(local)

                if self.args.enable_abs_comp is False:
                    threshold = percentile(local, value * 100)
                else:
                    threshold = percentile(local.abs(), value * 100)
                threshold_list.append(threshold)
            return threshold_list
        else:
            local = []
            for name, p in self.named_parameters():
                if hasattr(p, "is_score") and p.is_score:
                    local.append(p.detach().flatten())
            local = torch.cat(local)
            if self.args.enable_abs_comp is False:
                threshold = percentile(local, sparsity * 100)
            else:
                threshold = percentile(local.abs(), sparsity * 100)
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

    def forward(
        self,
        x,
        edge_index,
        sparsity=None,
        epoch=0,
        return_intermediate=False,
        layer_idx=None,
        edge_weight=None,
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
            before_sparsity = calculate_sparsity(x)
            print(f"before: {before_sparsity:.8f}")

            x_mask = self.get_feat_mask(
                x_pruning_ratio, self.x_weight, epoch=epoch
            )
            x = x * x_mask

            # x_maskを適用した後のsparsityを計算
            after_sparsity = calculate_sparsity(x)
            print(f"after: {after_sparsity:.8f}")

        for i in range(self.num_layers - 1):
            if return_intermediate:
                if i == layer_idx:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = self.layers_GCN[i](x, edge_index, threshold, sparsity)
                    if self.type_norm in ["batch", "pair"]:
                        x = self.layers_bn[i](x)
                    x = F.relu(x)
                    return x
                else:
                    pass
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.layers_GCN[i](x, edge_index, threshold, sparsity)
                if self.type_norm == "batch":
                    x = self.layers_bn[i](x)
                x = F.relu(x)

                if self.args.validate:
                    output_list = x.tolist()
                    with open(f"output_layer{i}.txt", "w") as file:
                        for item in output_list:
                            file.write("%s\n" % item)

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
                    before_sparsity = calculate_sparsity(x)
                    print(f"before: {before_sparsity:.8f}")

                    x_mask = self.get_feat_mask(
                        x_pruning_ratio, self.x_weight, epoch=epoch
                    )
                    x = x * x_mask

                    # x_maskを適用した後のsparsityを計算
                    after_sparsity = calculate_sparsity(x)
                    print(f"after: {after_sparsity:.8f}")

        if return_intermediate:
            if i + 1 == layer_idx:
                x = self.layers_GCN[-1](
                    x, edge_index, threshold, sparsity, epoch=epoch
                )
                return x
            else:
                pass
        else:
            x = self.layers_GCN[-1](
                x, edge_index, threshold, sparsity, epoch=epoch
            )
            if self.args.validate:
                output_list = x.tolist()
                with open(f"output_layer_last.txt", "w") as file:
                    for item in output_list:
                        file.write("%s\n" % item)
            return x

    def rerandomize(self, mode, la, mu, sparsity=None):
        for m in self.modules():
            if type(m) is GATConv:
                m.rerandomize(mode, la, mu, sparsity)
