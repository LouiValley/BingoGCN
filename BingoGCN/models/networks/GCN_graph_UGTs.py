import os

import numpy as np
import torch
import torch.nn.functional as F
from models.networks.sparse_modules import (
    GetSubnet,
    SparseLinear,
    SparseLinearMulti_mask,
    SparseModule,
)
from models.networks.sparse_modules_graph import GCNConv, Graphlevel_GCNConv
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch import nn
from torch_geometric.nn import global_mean_pool

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

# class AtomEncoder(torch.nn.Module):
#     def __init__(self, emb_dim):
#         super(AtomEncoder, self).__init__()

#         self.atom_embedding_list = torch.nn.ModuleList()

#         for i, dim in enumerate(full_atom_feature_dims):
#             emb = torch.nn.Embedding(dim, emb_dim)
#             torch.nn.init.xavier_uniform_(emb.weight.data)
#             self.atom_embedding_list.append(emb)

#     def forward(self, x):
#         x_embedding = 0
#         for i in range(x.shape[1]):
#             x_embedding += self.atom_embedding_list[i](x[:, i])

#         return x_embedding


class AtomEncoder(torch.nn.Module):
    r"""The atom encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = AtomEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        torch.Size([10, 16])
    """

    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        encoded_features = 0
        for i in range(x.shape[1]):
            encoded_features += self.atom_embedding_list[i](x[:, i])

        return encoded_features


class LT_AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(LT_AtomEncoder, self).__init__()

        self.atom_embedding_list = nn.ModuleList()
        full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            emb.weight.is_score = True
            emb.weight.sparsity = 0.0
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class SLT_AtomEncoder(nn.Module):

    def __init__(self, emb_dim, layer=1, args=None):
        super(SLT_AtomEncoder, self).__init__()
        if args.linear_sparsity is not None:
            self.sparsity = args.linear_sparsity
        else:
            self.sparsity = args.conv_sparsity

        self.atom_embedding_list = nn.ModuleList()
        full_atom_feature_dims = get_atom_feature_dims()
        self.weight_scores_list = nn.ParameterList()
        self.weight_zeros_list = []
        self.weight_ones_list = []

        self.args = args
        self.init_mode = args.init_mode
        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score
        self.sparsity_value = args.linear_sparsity
        self.enable_multi_mask = args.enable_mask
        self.enable_abs_comp = args.enable_abs_comp
        self.SLTAtom = args.SLTAtom
        self.SLTAtom_ini = args.SLTAtom_ini
        # self.SLTAtom_multi = args.SLTAtom_multi

        for i, dim in enumerate(full_atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            emb.weight.requires_grad = False
            weight_score = nn.Parameter(
                torch.ones(1, emb.weight.size(1))
            )  # build score as the same shape as weight
            weight_score.is_score = True
            weight_score.atom = True
            weight_score.sparsity = self.sparsity
            SparseModule.init_param_(
                self,
                weight_score,
                init_mode=self.init_mode_mask,
                scale=self.init_scale_score,
                layer=i,
                # list_num=i
            )
            weight_zeros = torch.zeros(1, emb.weight.size(1))
            weight_ones = torch.ones((1, emb.weight.size(1)))
            weight_zeros.requires_grad = False
            weight_ones.requires_grad = False
            # ...
            if self.SLTAtom_ini == "default":
                nn.init.xavier_uniform_(emb.weight.data)  # original
            else:
                SparseModule.init_param_(
                    self,
                    emb.weight,
                    init_mode=self.SLTAtom_ini,
                    scale=self.init_scale_score,
                    sparse_value=self.sparsity_value,
                    layer=i,
                )
            if self.args.validate and self.args.flowgnn_debug is True:
                # print()
                os.makedirs(
                    "./pretrained_model/Models/Atom",
                    exist_ok=True,
                )
                emb_weight = emb.weight.detach().cpu().numpy()
                # with open(
                #     f"./pretrained_model/Models/Atom/Atom_emb_weight{i+1}.bin",
                #     "wb",
                # ) as f:
                #     f.write(emb_weight.tobytes())
                emb_weight = emb_weight.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/Atom/Atom_emb_weight{i+1}.txt",
                    emb_weight,
                    delimiter=",",
                    fmt="%.6f",
                )
            emb.weight.requires_grad = False
            self.atom_embedding_list.append(emb)
            self.weight_scores_list.append(weight_score)
            self.weight_zeros_list.append(weight_zeros)
            self.weight_ones_list.append(weight_ones)

    def forward(self, x, threshold):
        x_embedding = 0
        if isinstance(threshold, float):
            threshold = [threshold]
        for i in range(x.shape[1]):
            weight_score = self.weight_scores_list[i]
            weight_zeros = self.weight_zeros_list[i]
            weight_ones = self.weight_ones_list[i]
            subnets = []
            if self.args.local_pruning is True:
                for threshold_v in threshold[i]:
                    if self.enable_abs_comp is False:
                        subnet = GetSubnet.apply(
                            weight_score,
                            threshold_v,
                            weight_zeros,
                            weight_ones,
                        )
                        subnets.append(subnet)
                    else:
                        subnet = GetSubnet.apply(
                            weight_score.abs(),
                            threshold_v,
                            weight_zeros,
                            weight_ones,
                        )
                        subnets.append(subnet)
            else:
                for threshold_v in threshold:
                    if self.enable_abs_comp is False:
                        subnet = GetSubnet.apply(
                            weight_score,
                            threshold_v,
                            weight_zeros,
                            weight_ones,
                        )
                        subnets.append(subnet)
                    else:
                        subnet = GetSubnet.apply(
                            weight_score.abs(),
                            threshold_v,
                            weight_zeros,
                            weight_ones,
                        )
                        subnets.append(subnet)
            combined_subnet = torch.stack(subnets).sum(dim=0)
            x_embedding += (
                self.atom_embedding_list[i](x[:, i]) * combined_subnet
            )

            if self.args.validate and self.args.flowgnn_debug is True:
                # print()
                os.makedirs(
                    "./pretrained_model/Models/Atom",
                    exist_ok=True,
                )
                combined_subnet = combined_subnet.detach().cpu().numpy()
                combined_subnet = combined_subnet.T
                combined_subnet = combined_subnet.astype(np.int32)
                with open(
                    f"./pretrained_model/Models/Atom/Atom_combined_subnet{i+1}.bin",
                    "wb",
                ) as f:
                    f.write(combined_subnet.tobytes())
                combined_subnet = combined_subnet.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/Atom/Atom_combined_subnet{i+1}.txt",
                    combined_subnet,
                    delimiter=",",
                    fmt="%d",
                )

                # atom_embedding_list = (
                #     self.atom_embedding_list.cpu().numpy()
                # )
                # atom_embedding_list = (
                #     self.atom_embedding_list.detach().cpu().numpy()
                # )
                # # 結果を保存するための1次元配列を初期化
                # result = np.zeros(
                #     atom_embedding_list.shape[0]
                #     * atom_embedding_list.shape[1],
                #     dtype=int,
                # )

                # # atom_embedding_listの値を1または0に変換して結果の配列に格納
                # index = 0
                # for i in range(atom_embedding_list.shape[1]):
                #     for j in range(atom_embedding_list.shape[0]):
                #         if atom_embedding_list[j, i] > 0:
                #             result[index] = 1
                #         else:
                #             result[index] = 0
                #         index += 1

                # # 結果をテキストファイルに保存
                # np.savetxt(
                #     f"./Atom_binary_atom_embedding_list{self.atom_embedding_list.shape}.txt",
                #     result,
                #     fmt="%d",
                # )

                # # Save as binary file
                # with open(
                #     f"./Atom_atom_embedding_list{self.atom_embedding_list.shape}.bin",
                #     "wb",
                # ) as f:
                #     f.write(atom_embedding_list.tobytes())

                # # Save as text file
                # np.savetxt(
                #     f"./Atom_atom_embedding_list{self.atom_embedding_list.shape}.txt",
                #     atom_embedding_list,
                #     delimiter=",",
                # )

        return x_embedding


def percentile(t, q):
    t_flat = t.view(-1)
    t_sorted, _ = torch.sort(t_flat)
    k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
    return t_sorted[k].item()


class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


def resetBN_is_score(tmpBN):
    tmpBN.weight.is_score = True
    tmpBN.bias.is_score = True
    tmpBN.weight.sparsity = 0.0
    tmpBN.bias.sparsity = 0.0
    return tmpBN


def reset_is_score(tmp):
    tmp.weight.is_score = True
    tmp.bias.is_score = True
    tmp.weight.sparsity = 0.0
    tmp.bias.sparsity = 0.0
    return tmp


def resetEMB_is_score(tmp):
    tmp.weight.is_score = True
    tmp.weight.sparsity = 0.0
    return tmp


class GCN_graph_UGTs(nn.Module):
    def __init__(self, args):
        super(GCN_graph_UGTs, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.args = args
        self.type_norm = args.type_norm
        self.num_feats = args.num_feats
        self.dim_hidden = args.dim_hidden
        self.num_layer = args.num_layers

        self.convs = nn.ModuleList([])

        if args.dataset == "hep10k":
            self.atom_encoder = torch.nn.Linear(
                4, self.dim_hidden, bias=True
            )
        elif args.train_mode == "score_only" and args.SLTAtom is True:
            self.atom_encoder = SLT_AtomEncoder(
                emb_dim=args.dim_hidden, args=args
            )
        # elif args.train_mode == "score_only" and args.SLTAtom is False:
        #     self.atom_encoder = LT_AtomEncoder(emb_dim=args.dim_hidden)
        else:
            self.atom_encoder = AtomEncoder(emb_dim=args.dim_hidden)

        if args.no_norm is False:
            self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            self.convs.append(
                Graphlevel_GCNConv(
                    self.dim_hidden,
                    self.dim_hidden,
                    cached=self.cached,
                    args=args,
                    layer=layer,
                )
            )
            if args.no_norm is False:
                if args.BN_track_running_stats is True:
                    if args.train_mode == "score_only":
                        self.batch_norms.append(
                            resetBN_is_score(
                                torch.nn.BatchNorm1d(args.dim_hidden)
                            )
                        )
                    else:
                        self.batch_norms.append(
                            torch.nn.BatchNorm1d(args.dim_hidden)
                        )
                else:
                    if args.train_mode == "score_only":
                        self.batch_norms.append(
                            resetBN_is_score(
                                torch.nn.BatchNorm1d(
                                    args.dim_hidden, track_running_stats=False
                                )
                            )
                        )
                    else:
                        self.batch_norms.append(
                            torch.nn.BatchNorm1d(
                                args.dim_hidden, track_running_stats=False
                            )
                        )

        # self.pool = global_add_pool
        self.pool = global_mean_pool
        # self.pool = global_max_pool

        if args.train_mode == "score_only" and not args.dense_for_last_layer:
            # チャネル数が1なので，NMSparsityを適応するのはできない．
            # if args.nmsparsity and args.enable_mask:
            #     self.lin = NMSparseMultiLinear(self.dim_hidden, 1, args=args, layer=layer)
            # elif args.nmsparsity:
            #     self.lin = NMSparseLinear(self.dim_hidden, 1, args=args, layer=layer)
            if args.dataset == "ogbg-molpcba":
                self.graph_pred_linear = SparseLinearMulti_mask(
                    self.dim_hidden, 128, args=args, layer=self.num_layers
                )
            elif args.enable_mask is True:
                self.graph_pred_linear = SparseLinearMulti_mask(
                    self.dim_hidden, 1, args=args, layer=self.num_layers
                )
            else:
                self.graph_pred_linear = SparseLinear(
                    self.dim_hidden, 1, args=args, layer=self.num_layers
                )

            # if args.enable_mask is True:
            #     self.graph_pred_linear = SparseLinearMulti_mask(
            #         self.dim_hidden, 1, args=args, layer=self.num_layers
            #     )
            # else:
            #     self.graph_pred_linear = SparseLinear(
            #         self.dim_hidden, 1, args=args, layer=self.num_layers
            #     )
        else:
            if args.dataset == "ogbg-molpcba":
                self.graph_pred_linear = torch.nn.Linear(
                    self.dim_hidden, 128, bias=True
                )
            else:
                self.graph_pred_linear = torch.nn.Linear(
                    self.dim_hidden, 1, bias=True
                )

        # self.graph_pred_linear = torch.nn.Linear(self.dim_hidden, 1, bias=False)

        self.enable_multi_mask = args.enable_mask

    def percentile(self, t, q):
        t_flat = t.view(-1)
        t_sorted, _ = torch.sort(t_flat)
        k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
        return t_sorted[k].item()

    def get_threshold(self, sparsity, epoch=None, i=None):
        if self.args.local_pruning is True:
            if self.enable_multi_mask is True:  # enable multi-mask
                threshold_list = []
                for value in sparsity:
                    local = []
                    for name, p in self.named_parameters():
                        if (
                            name != "graph_pred_linear.weight_score"
                            and hasattr(p, "is_score")
                            and not hasattr(p, "bond")
                            and not hasattr(p, "atom")
                            and not hasattr(p, "root")
                            and p.is_score
                            and p.sparsity == self.linear_sparsity
                            and f"{i}" in name
                        ):
                            local.append(p.detach().flatten())
                            # print(name)
                    local = torch.cat(local)
                    if self.enable_abs_comp is False:
                        threshold = self.percentile(local, value * 100)
                    else:
                        threshold = self.percentile(local.abs(), value * 100)
                    threshold_list.append(threshold)
                return threshold_list
            else:
                local = []
                for name, p in self.named_parameters():
                    if (
                        name != "graph_pred_linear.weight_score"
                        and hasattr(p, "is_score")
                        and not hasattr(p, "bond")
                        and not hasattr(p, "atom")
                        and "root" not in name
                        and p.is_score
                        and p.sparsity == self.linear_sparsity
                        and f"{i}" in name
                    ):
                        local.append(p.detach().flatten())
                        # print(name)
                local = torch.cat(local)
                # print(local.shape)
                if self.enable_abs_comp is False:
                    threshold = self.percentile(local, sparsity * 100)
                else:
                    threshold = self.percentile(local.abs(), sparsity * 100)
                return threshold
        else:
            if self.enable_multi_mask is True:  # enable multi-mask
                threshold_list = []
                for value in sparsity:
                    local = []
                    for name, p in self.named_parameters():
                        if (
                            name != "graph_pred_linear.weight_score"
                            and hasattr(p, "is_score")
                            and not hasattr(p, "bond")
                            and not hasattr(p, "atom")
                            and not hasattr(p, "root")
                            and p.is_score
                            and p.sparsity == self.linear_sparsity
                        ):
                            local.append(p.detach().flatten())
                    local = torch.cat(local)
                    # threshold=percentile(local,sparsity*100)
                    if self.enable_abs_comp is False:
                        threshold = self.percentile(local, value * 100)
                    else:
                        threshold = self.percentile(local.abs(), value * 100)
                    threshold_list.append(threshold)
                return threshold_list
            else:
                local = []
                for name, p in self.named_parameters():
                    if (
                        name != "graph_pred_linear.weight_score"
                        and hasattr(p, "is_score")
                        and not hasattr(p, "bond")
                        and not hasattr(p, "atom")
                        and not hasattr(p, "root")
                        and p.is_score
                        and p.sparsity == self.linear_sparsity
                    ):
                        local.append(p.detach().flatten())
                        # print(name)
                local = torch.cat(local)
                # print(local.shape)
                if self.enable_abs_comp is False:
                    threshold = self.percentile(local, sparsity * 100)
                else:
                    threshold = self.percentile(local.abs(), sparsity * 100)
                return threshold

    def atom_get_threshold(self, sparsity, epoch=None, i=None):
        if self.args.local_pruning is True:
            # 外側のリストの初期化
            outer_threshold_list = []

            # 各atom属性を持つ重みに対するループ (0から8まで)
            for atom_idx in range(9):
                atom_str = f".{atom_idx}"

                # 対象のatom属性を持つ重みだけをフィルタリング
                filtered_params = []
                for name, p in self.named_parameters():
                    if (
                        atom_str in name
                        and hasattr(p, "is_score")
                        and hasattr(p, "atom")
                        and p.is_score
                        and p.sparsity == self.linear_sparsity
                    ):
                        # 追加: 条件を満たすパラメータの名前を表示
                        # print(name)

                        filtered_params.append(p)

                if self.enable_multi_mask:  # enable multi-mask
                    threshold_list = []
                    for value in sparsity:
                        local = [p.detach().flatten() for p in filtered_params]
                        local = torch.cat(local)
                        if self.enable_abs_comp is False:
                            threshold = self.percentile(local, value * 100)
                        else:
                            threshold = self.percentile(
                                local.abs(), value * 100
                            )
                        threshold_list.append(threshold)
                    outer_threshold_list.append(threshold_list)
                else:
                    local = [p.detach().flatten() for p in filtered_params]
                    local = torch.cat(local)
                    if self.enable_abs_comp is False:
                        threshold = self.percentile(local, sparsity * 100)
                    else:
                        threshold = self.percentile(
                            local.abs(), sparsity * 100
                        )
                    outer_threshold_list.append([threshold])
                    # print(name)
            return outer_threshold_list
        else:
            if self.enable_multi_mask is True:
                threshold_list = []
                for value in sparsity:
                    local = []
                    for name, p in self.named_parameters():
                        if (
                            hasattr(p, "is_score")
                            and hasattr(p, "atom")
                            and p.is_score
                            and p.sparsity == self.linear_sparsity
                        ):
                            local.append(p.detach().flatten())
                    local = torch.cat(local)
                    if self.enable_abs_comp is False:
                        threshold = self.percentile(local, value * 100)
                    else:
                        threshold = self.percentile(local.abs(), value * 100)
                    threshold_list.append(threshold)
                return threshold_list
            else:
                local = []
                for name, p in self.named_parameters():
                    if (
                        hasattr(p, "is_score")
                        and hasattr(p, "atom")
                        and p.is_score
                        and p.sparsity == self.linear_sparsity
                    ):
                        local.append(p.detach().flatten())
                        # print(name)
                local = torch.cat(local)
                # print(local.shape)
                if self.enable_abs_comp is False:
                    threshold = self.percentile(local, sparsity * 100)
                else:
                    threshold = self.percentile(local.abs(), sparsity * 100)
                return threshold

    def pred_get_threshold(self, sparsity, epoch=None):
        if self.enable_multi_mask is True:  # enable multi-mask
            threshold_list = []
            for value in sparsity:
                local = []
                for name, p in self.named_parameters():
                    if (
                        name == "graph_pred_linear.weight_score"
                        and p.is_score
                        and p.sparsity == self.linear_sparsity
                    ):
                        local.append(p.detach().flatten())
                local = torch.cat(local)
                if self.enable_abs_comp is False:
                    threshold = self.percentile(local, value * 100)
                else:
                    threshold = self.percentile(local.abs(), value * 100)
                threshold_list.append(threshold)
            return threshold_list
        else:
            local = []
            for name, p in self.named_parameters():
                if (
                    name == "graph_pred_linear.weight_score"
                    and p.is_score
                    and p.sparsity == self.linear_sparsity
                ):
                    local.append(p.detach().flatten())
                    # print(name)
            local = torch.cat(local)
            # print(local.shape)
            if self.enable_abs_comp is False:
                threshold = self.percentile(local, sparsity * 100)
            else:
                threshold = self.percentile(local.abs(), sparsity * 100)
            return threshold

    def forward(self, batched_data, sparsity, epoch):
        # when eval fixed sparsity
        if sparsity is None:
            if self.args.local_pruning:
                sparsity = self.args.local_sparsity_list
            elif self.args.enable_mask:
                sparsity = self.args.sparsity_list
            else:
                sparsity = self.args.linear_sparsity

        if (
            self.args.train_mode == "score_only"
            and self.args.local_pruning is False
        ):
            if not self.args.nmsparsity:
                threshold = self.get_threshold(sparsity, epoch=epoch)
            if self.args.SLTAtom is True:
                atom_threshold = self.atom_get_threshold(sparsity, epoch=epoch)
            if not self.args.dense_for_last_layer:
                pred_threshold = self.pred_get_threshold(sparsity, epoch=epoch)

        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        if self.args.validate is True:
            # Get the total number of nodes from the edge_index
            num_nodes = edge_index.max().item() + 1

            # Calculate the total number of elements in the adjacency matrix
            total_elements = num_nodes * num_nodes

            # Non-zero elements are just the number of edges
            non_zero_elements = edge_index.shape[1]

            # Calculate the number of zero elements
            zero_elements = total_elements - non_zero_elements

            # Calculate the sparsity (zero ratio)
            zero_ratio = zero_elements / total_elements

            # Output the computed statistics
            print(
                f"A:{zero_elements}, non_zero:{non_zero_elements}, "
                f"zero_ratio:{zero_ratio}, "
                f"shape:torch.Size([{num_nodes}, {num_nodes}])"
            )
#
#         if self.args.dataset == "hep10k":
#             h = x
        if self.args.SLTAtom is True:
            if (
                self.args.train_mode == "score_only"
                and self.args.local_pruning is True
            ):
                atom_threshold = self.atom_get_threshold(
                    sparsity[-2], epoch=epoch
                )
                # h_list = [self.atom_encoder(x, atom_threshold)]
                h = self.atom_encoder(x, atom_threshold)
            elif (
                self.args.train_mode == "score_only"
                and self.args.local_pruning is False
            ):
                if self.args.flowgnn_debug:
                    os.makedirs(
                        "./pretrained_model/Output/Atom",
                        exist_ok=True,
                    )
                    before_atom_x = x.detach().cpu().numpy()
                    before_atom_x = before_atom_x.T
                    with open(
                        "./pretrained_model/Output/Atom/before_atom_x.bin",
                        "wb",
                    ) as f:
                        f.write(before_atom_x.tobytes())
                    before_atom_x = before_atom_x.ravel(order="F")
                    np.savetxt(
                        "./pretrained_model/Output/Atom/before_atom_x.txt",
                        before_atom_x,
                        fmt="%d",
                        delimiter=",",
                    )
                # h_list = [self.atom_encoder(x, atom_threshold)]
                h = self.atom_encoder(x, atom_threshold)
                if self.args.flowgnn_debug:
                    os.makedirs(
                        "./pretrained_model/Output/Atom",
                        exist_ok=True,
                    )
                    # after_atom_x = h_list[0].detach().cpu().numpy()
                    after_atom_x = h.detach().cpu().numpy()
                    after_atom_x = after_atom_x.T
                    with open(
                        "./pretrained_model/Output/Atom/after_atom_x.bin",
                        "wb",
                    ) as f:
                        f.write(after_atom_x.tobytes())
                    after_atom_x = after_atom_x.ravel(order="F")
                    np.savetxt(
                        "./pretrained_model/Output/Atom/after_atom_x.txt",
                        after_atom_x,
                        fmt="%.6f",
                        delimiter=",",
                    )
        else:
            # h_list = [self.atom_encoder(x)]
            h = self.atom_encoder(x)
            if self.args.flowgnn_debug:
                all_atom_embs = []

                for i in range(9):
                    print_atom_emb = (
                        getattr(self.atom_encoder.atom_embedding_list, str(i))
                        .weight.detach()
                        .cpu()
                        .numpy()
                    )
                    print_atom_emb = print_atom_emb.T
                    all_atom_embs.append(print_atom_emb)

                # 結合して保存
                all_atom_embs = np.concatenate(all_atom_embs, axis=1)
                all_atom_embs = all_atom_embs.T
                shape = all_atom_embs.shape

                os.makedirs(
                    "./pretrained_model/Models/Atom",
                    exist_ok=True,
                )

                with open(
                    f"./pretrained_model/Models/Atom/atom_emb_combined_{shape[0]}x{shape[1]}.bin",
                    "wb",
                ) as f:
                    f.write(all_atom_embs.tobytes())

                all_atom_embs = all_atom_embs.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/Atom/atom_emb_combined_{shape[0]}x{shape[1]}.txt",
                    all_atom_embs,
                    fmt="%.6f",
                    delimiter=",",
                )
        for layer in range(self.num_layer):

            # hのスパース性を計算 (非ゼロ要素の割合)
            # sparsity_ratio = (h == 0).sum().item() / h.numel()
            # print(f"Layer {layer}: h sparsity = {sparsity_ratio:.4f}")


            if self.args.validate:
                num_zeros = (x == 0).sum().item()
                num_elements = x.numel()
                sparsity_value = num_zeros / num_elements
                print(
                    f"Input_of_Layer{layer} before embedding x sparsity: {sparsity_value:.8f}"
                )

            if self.args.local_pruning is True:
                threshold = self.get_threshold(
                    sparsity[layer], epoch=epoch, i=layer
                )

            if self.args.nmsparsity:
                threshold = None
                h = self.convs[layer](
                    # h_list[layer],
                    h,
                    edge_index,
                    edge_attr,
                    threshold,
                    sparsity=sparsity,
                    batched_data=batched_data,
                )
            else:
                if (
                    self.args.train_mode == "score_only"
                    and self.args.local_pruning is True
                ):
                    h = self.convs[layer](
                        # h_list[layer],
                        h,
                        edge_index,
                        edge_attr,
                        threshold,
                        sparsity=sparsity[layer],
                        layer=layer,
                    )
                elif (
                    self.args.train_mode == "score_only"
                    and self.args.local_pruning is False
                ):
                    h = self.convs[layer](
                        # h_list[layer],
                        h,
                        edge_index,
                        edge_attr,
                        threshold,
                        sparsity=sparsity,
                    )
                else:
                    h = self.convs[layer](
                        # h_list[layer],
                        h,
                        edge_index,
                        edge_attr,
                        threshold=None,
                        sparsity=sparsity,
                    )

            if self.args.flowgnn_debug:
                # Calculate sparsity for each data point in the encoded batch
                encoded_sparsity = []
                for i in range(batched_data.num_graphs):
                    mask = batch == i
                    h_i = h[mask]  # Get the encoded nodes for the i-th graph
                    num_elements_i = h_i.numel()
                    non_zero_elements_i = (h_i != 0).sum().item()
                    zero_elements_i = num_elements_i - non_zero_elements_i
                    zero_ratio_i = zero_elements_i / num_elements_i
                    encoded_sparsity.append(zero_ratio_i)

                # print(f"AXW sparsity: {encoded_sparsity}")

            if self.args.no_norm is False:
                h = self.batch_norms[layer](h)
                if self.args.validate and self.args.flowgnn_debug is True:
                    weight = (
                        self.batch_norms[layer].weight.detach().cpu().numpy()
                    )
                    bias = self.batch_norms[layer].bias.detach().cpu().numpy()
                    mean = (
                        self.batch_norms[layer]
                        .running_mean.detach()
                        .cpu()
                        .numpy()
                    )
                    var = (
                        self.batch_norms[layer]
                        .running_var.detach()
                        .cpu()
                        .numpy()
                    )
                    # sqrt_var = np.sqrt(var + 1e-5)

                    os.makedirs(
                        f"./pretrained_model/Models/BN/l{layer+1}",
                        exist_ok=True,
                    )
                    base_path = f"./pretrained_model/Models/BN/l{layer+1}"

                    # 重みをテキストファイルに保存
                    np.savetxt(
                        f"{base_path}/l{layer+1}_bn_weight.txt",
                        weight,
                        delimiter=",",
                        fmt="%.6f",
                    )
                    # 重みをバイナリファイルに保存
                    with open(
                        f"{base_path}/l{layer+1}_bn_weight.bin",
                        "wb",
                    ) as f:
                        f.write(weight.tobytes())

                    # バイアスをテキストファイルに保存
                    np.savetxt(
                        f"{base_path}/l{layer+1}_bn_bias.txt",
                        bias,
                        delimiter=",",
                        fmt="%.6f",
                    )
                    # バイアスをバイナリファイルに保存
                    with open(
                        f"{base_path}/l{layer+1}_bn_bias.bin", "wb"
                    ) as f:
                        f.write(bias.tobytes())

                    # meanをテキストファイルに保存
                    np.savetxt(
                        f"{base_path}/l{layer+1}_bn_mean.txt",
                        mean,
                        delimiter=",",
                        fmt="%.6f",
                    )
                    # meanをバイナリファイルに保存
                    with open(
                        f"{base_path}/l{layer+1}_bn_mean.bin",
                        "wb",
                    ) as f:
                        f.write(mean.tobytes())

                    # sqrt(var)をテキストファイルに保存
                    # np.savetxt(
                    #     f"{base_path}/l{layer+1}_bn_sqrt_var.txt",
                    #     sqrt_var,
                    #     delimiter=",",
                    #     fmt="%.6f",
                    # )
                    # # sqrt(var)をバイナリファイルに保存
                    # with open(
                    #     f"{base_path}/l{layer+1}_bn_sqrt_var.bin",
                    #     "wb",
                    # ) as f:
                    #     f.write(sqrt_var.tobytes())

                    # sqrt(var)をテキストファイルに保存
                    np.savetxt(
                        f"{base_path}/l{layer+1}_bn_var.txt",
                        var,
                        delimiter=",",
                        fmt="%.6f",
                    )
                    # sqrt(var)をバイナリファイルに保存
                    with open(
                        f"{base_path}/l{layer+1}_bn_var.bin",
                        "wb",
                    ) as f:
                        f.write(var.tobytes())

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.args.dropout, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.args.dropout, training=self.training
                )

            if self.args.validate:
                output_list = h.tolist()
                with open(f"Output of Layer {layer}.txt", "w") as file:
                    for item in output_list:
                        file.write("%s\n" % item)

            # h_list.append(h)

        # h_node = h_list[-1]

        # print(h_node.shape)
        h_graph = self.pool(h, batch)

        if self.args.flowgnn_debug:
            os.makedirs(
                "./pretrained_model/Output/Pool",
                exist_ok=True,
            )
            after_pool = h_graph.detach().cpu().numpy()
            after_pool = after_pool.T
            with open(
                "./pretrained_model/Output/Pool/after_pool.bin",
                "wb",
            ) as f:
                f.write(after_pool.tobytes())
            after_pool = after_pool.ravel(order="F")
            np.savetxt(
                "./pretrained_model/Output/Pool/after_pool.txt",
                after_pool,
                fmt="%.6f",
                delimiter=",",
            )

        if (
            self.args.train_mode == "score_only"
            and self.args.local_pruning is True
        ):
            pred_threshold = self.pred_get_threshold(sparsity[-1], epoch=epoch)
            x = self.graph_pred_linear(h_graph, pred_threshold)
        elif (
            self.args.train_mode == "score_only"
            and not self.args.dense_for_last_layer
        ):
            x = self.graph_pred_linear(h_graph, pred_threshold)
        else:
            x = self.graph_pred_linear(h_graph)
            if self.args.flowgnn_debug:
                os.makedirs(
                    "./pretrained_model/Output/Pred",
                    exist_ok=True,
                )
                after_pred_x = x.detach().cpu().numpy()
                after_pred_x = after_pred_x.T
                with open(
                    "./pretrained_model/Output/Pred/after_pred_x.bin",
                    "wb",
                ) as f:
                    f.write(after_pred_x.tobytes())
                after_pred_x = after_pred_x.ravel(order="F")
                np.savetxt(
                    "./pretrained_model/Output/Pred/after_pred_x.txt",
                    after_pred_x,
                    fmt="%.6f",
                    delimiter=",",
                )

                # 重みとバイアスを取得
                os.makedirs(
                    "./pretrained_model/Models/WandB/pred",
                    exist_ok=True,
                )
                weight = self.graph_pred_linear.weight.detach().cpu().numpy()
                bias = self.graph_pred_linear.bias.detach().cpu().numpy()

                # 重みをバイナリファイルとして保存
                with open(
                    f"./pretrained_model/Models/WandB/pred/pred_weight_{weight.shape}.bin",
                    "wb",
                ) as f:
                    f.write(weight.tobytes())

                # 重みをテキストファイルとして保存
                weight_flat = weight.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/WandB/pred/pred_weight_{weight.shape}.txt",
                    weight_flat,
                    delimiter=",",
                    fmt="%.6f",
                )

                # バイアスをバイナリファイルとして保存
                with open(
                    f"./pretrained_model/Models/WandB/pred/pred_bias_{bias.shape}.bin",
                    "wb",
                ) as f:
                    f.write(bias.tobytes())

                # バイアスをテキストファイルとして保存
                bias_flat = bias.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/WandB/pred/pred_bias_{bias.shape}.txt",
                    bias_flat,
                    delimiter=",",
                    fmt="%.6f",
                )

        return x

    def rerandomize(self, mode, la, mu):
        for m in self.modules():
            if type(m) is GCNConv:
                m.rerandomize(mode, la, mu)
