import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from torch_geometric.nn import GCNConv
from .sparse_modules_graph import GCNConv

# def percentile(t, q):
#     t_flat = t.view(-1)
#     t_sorted, _ = torch.sort(t_flat)
#     k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
#     return t_sorted[k].item()


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


def save_bn_parameters(layer_bn=None, layer_idx=None, num_layer=None):
    os.makedirs(f"./pretrained_model/Models/BN/l{layer_idx+1}", exist_ok=True)
    base_path = f"./pretrained_model/Models/BN/l{layer_idx+1}"

    bn_weight = layer_bn.weight.data.cpu().numpy()
    bn_bias = layer_bn.bias.data.data.cpu().numpy()
    bn_running_mean = layer_bn.running_mean.data.cpu().numpy()
    bn_running_var = layer_bn.running_var.data.cpu().numpy()
    if layer_idx + 1 == num_layer:
        bn_running_sqrt_var = bn_running_var
    else:
        bn_running_sqrt_var = np.sqrt(bn_running_var + 1e-5)

    with open(
        f"{base_path}/l{layer_idx+1}_bn_weight.bin",
        "wb",
    ) as f:
        f.write(bn_weight.tobytes())
    with open(
        f"{base_path}/l{layer_idx+1}_bn_bias.bin",
        "wb",
    ) as f:
        f.write(bn_bias.tobytes())
    with open(
        f"{base_path}/l{layer_idx+1}_bn_mean.bin",
        "wb",
    ) as f:
        f.write(bn_running_mean.tobytes())
    with open(
        f"{base_path}/l{layer_idx+1}_bn_sqrt_var.bin",
        "wb",
    ) as f:
        f.write(bn_running_sqrt_var.tobytes())

    # torch.save(
    #     layer_bn.weight.data, f"{base_path}/l{layer_idx+1}_bn_weight.bin"
    # )
    # torch.save(layer_bn.bias.data, f"{base_path}/l{layer_idx+1}_bn_bias.bin")
    # torch.save(
    #     layer_bn.running_mean, f"{base_path}/l{layer_idx+1}_bn_mean.bin"
    # )
    # torch.save(layer_bn.running_var, f"{base_path}/l{layer_idx+1}_bn_var.bin")

    np.savetxt(
        f"{base_path}/l{layer_idx+1}_bn_weight.txt",
        layer_bn.weight.data.cpu().numpy(),
    )
    np.savetxt(
        f"{base_path}/l{layer_idx+1}_bn_bias.txt",
        layer_bn.bias.data.cpu().numpy(),
    )
    np.savetxt(
        f"{base_path}/l{layer_idx+1}_bn_mean.txt",
        layer_bn.running_mean.cpu().numpy(),
    )
    np.savetxt(
        f"{base_path}/l{layer_idx+1}_bn_sqrt_var.txt",
        bn_running_sqrt_var,
    )


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        # k_val = percentile(scores, sparsity*100)
        # if glob:
        out = torch.where(scores < threshold, zeros.to(scores.device), ones.to(scores.device))
        # else:
        #    k_val = percentile(scores, threshold*100)
        #    out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.args = args
        self.dropout_rate = args.dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layers_GCN.append(
            GCNConv(
                self.num_feats,
                self.dim_hidden,
                cached=self.cached,
                args=args,
                layer=0,
            )
        )
        if self.type_norm == "batch":
            if args.train_mode == "score_only":
                self.layers_bn.append(
                    # resetBN_is_score(
                    torch.nn.BatchNorm1d(
                        self.dim_hidden,
                        momentum=args.bn_momentum,
                        track_running_stats=args.bn_track_running_stats,
                        affine=args.bn_affine,
                    )
                    # )
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
        elif self.type_norm == "layer":
            if args.train_mode == "score_only":
                self.layers_bn.append(
                    resetBN_is_score(
                        torch.nn.LayerNorm(
                            self.dim_hidden,
                            elementwise_affine=args.bn_affine,
                        )
                    )
                )
            else:
                self.layers_bn.append(
                    torch.nn.LayerNorm(
                        self.dim_hidden,
                        elementwise_affine=args.bn_affine,
                    )
                )
        elif self.type_norm == "pair":
            self.layers_bn.append(pair_norm())

        for i in range(self.num_layers - 2):
            self.layers_GCN.append(
                GCNConv(
                    self.dim_hidden,
                    self.dim_hidden,
                    cached=self.cached,
                    args=args,
                    layer=i + 1,
                )
            )

            if self.type_norm == "batch":
                if args.train_mode == "score_only":
                    self.layers_bn.append(
                        # resetBN_is_score(
                        torch.nn.BatchNorm1d(
                            self.dim_hidden,
                            momentum=args.bn_momentum,
                            track_running_stats=args.bn_track_running_stats,
                            affine=args.bn_affine,
                        )
                        # )
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
            elif self.type_norm == "layer":
                if args.train_mode == "score_only":
                    self.layers_bn.append(
                        resetBN_is_score(
                            torch.nn.LayerNorm(
                                self.dim_hidden,
                                elementwise_affine=args.bn_affine,
                            )
                        )
                    )
                else:
                    self.layers_bn.append(
                        torch.nn.LayerNorm(
                            self.dim_hidden,
                            elementwise_affine=args.bn_affine,
                        )
                    )
            elif self.type_norm == "pair":
                self.layers_bn.append(pair_norm())

        self.layers_GCN.append(
            GCNConv(
                self.dim_hidden,
                self.num_classes,
                cached=self.cached,
                args=args,
                layer=self.num_layers - 1,
            )
        )

        # if self.type_norm == "batch":
        #     if args.train_mode == "score_only":
        #         self.layers_bn.append(
        #             # resetBN_is_score(
        #             torch.nn.BatchNorm1d(
        #                 self.dim_hidden,
        #                 momentum=args.bn_momentum,
        #                 track_running_stats=args.bn_track_running_stats,
        #                 affine=args.bn_affine,
        #             )
        #             # )
        #         )
        #     else:
        #         self.layers_bn.append(
        #             torch.nn.BatchNorm1d(
        #                 self.dim_hidden,
        #                 momentum=args.bn_momentum,
        #                 track_running_stats=args.bn_track_running_stats,
        #                 affine=args.bn_affine,
        #             )
        #         )
        # elif self.type_norm == "layer":
        #     if args.train_mode == "score_only":
        #         self.layers_bn.append(
        #             resetBN_is_score(
        #                 torch.nn.LayerNorm(
        #                     self.dim_hidden,
        #                     elementwise_affine=args.bn_affine,
        #                 )
        #             )
        #         )
        #     else:
        #         self.layers_bn.append(
        #             torch.nn.LayerNorm(
        #                 self.dim_hidden,
        #                 elementwise_affine=args.bn_affine,
        #             )
        #         )
        # elif self.type_norm == "pair":
        #     self.layers_bn.append(pair_norm())

        # self.optimizer = torch.optim.Adam(self.parameters(),
        #                                 lr=self.lr, weight_decay=self.weight_decay)

        if self.args.enable_feat_pruning:
            if self.args.x_pruning_layer == 0:
                self.x_weight = nn.Parameter(torch.ones(1, self.num_feats))
            else:
                self.x_weight = nn.Parameter(torch.ones(1, self.args.dim_hidden))

            self.init_param_(
                self.x_weight,
                init_mode=self.init_mode_mask,
                scale=self.init_scale_score,
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
                init_mode=self.init_mode_mask,
                scale=self.init_scale_score,
            )
            self.x_weight_zeros = torch.zeros(self.x_weight.size())
            self.x_weight_ones = torch.ones(self.x_weight.size())
            self.x_weight.is_score = True
            self.x_weight.sparsity = self.args.featsparsity_ratio
            self.x_weight.requires_grad = True
            self.x_weight_zeros.requires_grad = False
            self.x_weight_ones.requires_grad = False

    def init_param_(self, param, init_mode=None, scale=None, sparse_value=None):
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
        elif init_mode == "signed_constant_sparse" or init_mode == "signed_constant_SF":
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
                        and p.is_weight_score
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
                if self.args.train_mode == "score_only":
                    if (
                        hasattr(p, "is_score")
                        and p.is_score
                        and p.sparsity == self.args.linear_sparsity
                        and p.is_weight_score
                    ):
                        local.append(p.detach().flatten())
                else:
                    if (
                        hasattr(p, "is_score")
                        and p.is_score
                        and p.sparsity == self.args.linear_sparsity
                    ):
                        local.append(p.detach().flatten())
            local = torch.cat(local)
            if self.args.enable_abs_comp is False:
                threshold = percentile(local, sparsity * 100)
            else:
                threshold = percentile(local.abs(), sparsity * 100)
            return threshold

    def get_feat_mask(self, sparsity, x_weight, epoch=None):
        # local = []
        # local.append(x_weight.detach().flatten())
        # local = torch.cat(local)
        # threshold = torch.quantile(x_weight.abs(), sparsity)

        local = []
        local.append(x_weight.detach().flatten())
        local = torch.cat(local)
        threshold = percentile(local.abs(), sparsity * 100)
        mask = GetSubnet.apply(x_weight.abs(), threshold, self.x_weight_zeros, self.x_weight_ones)
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
        epoch=None,
        return_intermediate=False,
        layer_idx=None,
        edge_weight=None,
        part=None,
    ):
        if sparsity is None:
            if self.args.enable_mask is True:
                sparsity = self.args.sparsity_list
            else:
                sparsity = self.args.linear_sparsity

        if self.args.train_mode == "normal":
            threshold = self.get_threshold(sparsity, epoch=epoch)
        elif self.args.train_mode == "score_only":
            if self.args.nmsparsity is False:
                threshold = self.get_threshold(sparsity, epoch=epoch)
            else:
                threshold = None

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
            before_sparsity = calculate_sparsity(x)
            print(f"before: {before_sparsity:.8f}")
            x_mask = self.get_feat_mask(x_pruning_ratio, self.x_weight, epoch=epoch)
            x = x * x_mask
            after_sparsity = calculate_sparsity(x)
            print(f"after: {after_sparsity:.8f}")

        for i in range(self.num_layers - 1):
            if return_intermediate:
                if i == layer_idx:
                    if self.args.original_edge_weight:
                        x = self.layers_GCN[i](
                            x,
                            edge_index,
                            threshold,
                            sparsity,
                            None,
                            args=self.args,
                        )
                        if self.type_norm in ["batch", "pair"]:
                            x = self.layers_bn[i](x)
                        if not self.args.kmeans_before_relu:
                            x = F.relu(x)
                    else:
                        x = self.layers_GCN[i](
                            x,
                            edge_index,
                            threshold,
                            sparsity,
                            edge_weight,
                            args=self.args,
                            part=part,
                        )
                        if self.type_norm in ["batch", "pair"]:
                            x = self.layers_bn[i](x)
                        if not self.args.kmeans_before_relu:
                            x = F.relu(x, inplace=True)
                    return x
                else:
                    pass
            else:
                # if self.args.sparsity_profiling:
                #     num_zeros = (x == 0).sum().item()
                #     num_elements = x.numel()
                #     sparsity_value = num_zeros / num_elements
                #     print(f"Layer {i} x sparsity: {sparsity_value:.8f}")
                x = self.layers_GCN[i](x, edge_index, threshold, sparsity, args=self.args)
                if self.type_norm in ["batch", "pair"]:
                    x = self.layers_bn[i](x)
                    if self.args.flowgnn_debug and self.args.train_mode == "normal":
                        save_bn_parameters(self.layers_bn[i], i, self.args.num_layers)

                # if self.args.sparsity_profiling:
                #     num_zeros = (x == 0).sum().item()
                #     num_elements = x.numel()
                #     sparsity_value = num_zeros / num_elements
                #     print(f"AXW sparsity: {sparsity_value:.8f}")

                x = F.relu(x)
                if self.training:  # 訓練時のみDropoutを適用
                    x = self.dropout(x)

                if self.args.x_pruning_layer - 1 == i and (
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
                    before_sparsity = calculate_sparsity(x)
                    print(f"before: {before_sparsity:.8f}")
                    x_mask = self.get_feat_mask(x_pruning_ratio, self.x_weight, epoch=epoch)
                    x = x * x_mask
                    after_sparsity = calculate_sparsity(x)
                    print(f"after: {after_sparsity:.8f}")

                # if self.args.validate:
                #     os.makedirs("./pretrained_model/Output/", exist_ok=True)
                #     output_list = x.tolist()
                #     with open(
                #         f"./pretrained_model/Output/output_layer{i}.txt", "w"
                #     ) as file:
                #         for item in output_list:
                #             file.write("%s\n" % item)

        if return_intermediate:
            if i + 1 == layer_idx:
                if self.args.original_edge_weight:
                    x = self.layers_GCN[-1](
                        x,
                        edge_index,
                        threshold,
                        sparsity,
                        None,
                        args=self.args,
                    )
                else:
                    x = self.layers_GCN[-1](
                        x,
                        edge_index,
                        threshold,
                        sparsity,
                        edge_weight,
                        args=self.args,
                    )
                return x
            else:
                pass
        else:

            # if self.args.sparsity_profiling:
            #     num_zeros = (x == 0).sum().item()
            #     num_elements = x.numel()
            #     sparsity_value = num_zeros / num_elements
            #     print(f"Layer last x sparsity_value: {sparsity_value:.8f}")

            x = self.layers_GCN[-1](x, edge_index, threshold, sparsity, args=self.args, epoch=epoch)

            if self.args.flowgnn_debug:
                if self.args.train_mode == "normal":
                    layer_bn = torch.nn.BatchNorm1d(
                        self.dim_hidden,
                        momentum=self.args.bn_momentum,
                        track_running_stats=self.args.bn_track_running_stats,
                        affine=self.args.bn_affine,
                    )
                    save_bn_parameters(layer_bn, self.num_layers - 1, self.args.num_layers)

            if self.args.validate:
                output_list = x.tolist()
                with open(
                    f"./pretrained_model/Output/output_layer{self.num_layers-1}.txt",
                    "w",
                ) as file:
                    for item in output_list:
                        file.write("%s\n" % item)

            # if self.args.sparsity_profiling:
            #     num_zeros = (x == 0).sum().item()
            #     num_elements = x.numel()
            #     sparsity = num_zeros / num_elements
            #     print(f"AXW sparsity: {sparsity:.8f}")

            if self.args.dataset == "Reddit":
                x = F.log_softmax(x, dim=1)

            return x

    def rerandomize(self, mode, la, mu):
        for m in self.modules():
            if type(m) is GCNConv:
                m.rerandomize(mode, la, mu)
