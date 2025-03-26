import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    degree,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor, matmul, set_diag
from torch_sparse import sum as sparsesum

from .sparse_modules import (
    GetSubnet,
    NMSparseLinear,
    NMSparseMultiLinear,
    SparseLinear,
    SparseLinearMulti_mask,
    SparseModule,
    SparseParameter,
    SparseParameterMulti_mask,
)


def resetEMB_is_score(tmp):
    tmp.weight.is_score = True
    tmp.weight.sparsity = 0.0
    return tmp


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        full_bond_feature_dims = get_bond_feature_dims()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class SLT_BondEncoder(nn.Module):
    def __init__(self, emb_dim, layer=1, args=None):
        super(SLT_BondEncoder, self).__init__()
        # full_atom_feature_dims = get_atom_feature_dims()
        full_bond_feature_dims = get_bond_feature_dims()
        if args.linear_sparsity is not None:
            self.sparsity = args.linear_sparsity
        else:
            self.sparsity = args.conv_sparsity
        self.bond_embedding_list = nn.ModuleList()
        self.weight_scores_list = nn.ParameterList()
        self.weight_zeros_list = []
        self.weight_ones_list = []
        full_bond_feature_dims = get_bond_feature_dims()
        self.args = args
        self.init_mode = args.init_mode
        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score
        self.sparsity_value = args.linear_sparsity
        self.enable_multi_mask = args.enable_mask
        self.enable_abs_comp = args.enable_abs_comp
        self.enable_sw_mm = args.enable_sw_mm
        self.layer = layer
        self.SLTBond_ini = args.SLTBond_ini
        # self.SLTBond_multi = args.SLTBond_multi

        for i, dim in enumerate(full_bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            emb.weight.requires_grad = False
            if self.enable_sw_mm is True:
                for j in range(self.layer):
                    weight_score = nn.Parameter(
                        torch.ones(1, emb.weight.size(1))
                    )  # build score as the same shape as weight
                    weight_score.is_score = True
                    weight_score.bond = True
                    weight_score.sparsity = self.sparsity
                    SparseModule.init_param_(
                        self,
                        weight_score,
                        init_mode=self.init_mode_mask,
                        scale=self.init_scale_score,
                        layer=layer * 10 + i,
                    )
                    self.weight_scores_list.append(weight_score)
            else:
                weight_score = nn.Parameter(
                    torch.ones(1, emb.weight.size(1))
                )  # build score as the same shape as weight
                weight_score.is_score = True
                weight_score.bond = True
                weight_score.sparsity = self.sparsity
                SparseModule.init_param_(
                    self,
                    weight_score,
                    init_mode=self.init_mode_mask,
                    scale=self.init_scale_score,
                    layer=layer * 10 + i,
                )
                self.weight_scores_list.append(weight_score)
            # weight_zeros = torch.zeros(emb.weight.size())
            # weight_ones = torch.ones(emb.weight.size())
            weight_zeros = torch.zeros(1, emb.weight.size(1))
            weight_ones = torch.ones((1, emb.weight.size(1)))
            weight_zeros.requires_grad = False
            weight_ones.requires_grad = False
            # ..
            if self.SLTBond_ini == "default":
                nn.init.xavier_uniform_(emb.weight.data)  # original
            else:
                SparseModule.init_param_(
                    self,
                    emb.weight,
                    init_mode=self.SLTBond_ini,
                    scale=self.init_scale_score,
                    sparse_value=self.sparsity_value,
                    layer=layer,
                    list_num=i,
                )

            self.bond_embedding_list.append(emb)
            self.weight_zeros_list.append(weight_zeros)
            self.weight_ones_list.append(weight_ones)
            if self.args.validate and self.args.flowgnn_debug is True:
                # print()
                os.makedirs(
                    f"./pretrained_model/Models/Bond/L{self.layer}",
                    exist_ok=True,
                )
                emb_weight = emb.weight.detach().cpu().numpy()
                emb_weight = emb_weight.T
                # with open(
                #     f"./pretrained_model/Models/Bond/Bond_emb_weight{i+1}.bin",
                #     "wb",
                # ) as f:
                #     f.write(emb_weight.tobytes())
                emb_weight = emb_weight.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/Bond/L{self.layer}/Bond_emb_weight{i+1}.txt",
                    emb_weight,
                    delimiter=",",
                    fmt="%.6f",
                )

    def forward(self, edge_attr, threshold, index_mask=0):
        bond_embedding = 0
        if isinstance(threshold, float):
            threshold = [threshold]
        for i in range(edge_attr.shape[1]):
            if self.enable_sw_mm is True:
                weight_score = self.weight_scores_list[i * self.layer + index_mask]
            else:
                weight_score = self.weight_scores_list[i]
            weight_zeros = self.weight_zeros_list[i]
            weight_ones = self.weight_ones_list[i]
            subnets = []
            if self.args.local_pruning is True:
                # iの追加
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
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i]) * combined_subnet
            if self.args.validate and self.args.flowgnn_debug is True:

                os.makedirs(
                    f"./pretrained_model/Models/Bond/L{self.layer}",
                    exist_ok=True,
                )
                combined_subnet = combined_subnet.detach().cpu().numpy()
                combined_subnet = combined_subnet.T
                combined_subnet = combined_subnet.astype(np.int32)
                with open(
                    f"./pretrained_model/Models/Bond/L{self.layer}/Bond_combined_subnet{i+1}.bin",
                    "wb",
                ) as f:
                    f.write(combined_subnet.tobytes())

                combined_subnet = combined_subnet.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/Bond/L{self.layer}/Bond_combined_subnet{i+1}.txt",
                    combined_subnet,
                    delimiter=",",
                    fmt="%d",
                )

        return bond_embedding


def percentile(t, q):
    t_flat = t.view(-1)
    t_sorted, _ = torch.sort(t_flat)
    k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
    return t_sorted[k].item()


# def percentile(t, q):
#     t_flat = t.view(-1)
#     t_sorted, _ = torch.sort(t_flat)
#     k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
#     return t_sorted[k].item()


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)
    or
    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),
    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        args=None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.args = args
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        threshold=None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out, threshold)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


# @torch.jit._overload
# def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
#     # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
#     pass


# @torch.jit._overload
# def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
#     # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
#     pass


def add_remaining_self_loops_sparse(
    edge_index: Union[Tensor, SparseTensor],
    edge_attr: OptTensor = None,
    fill_value: Union[float, Tensor, str] = None,
    num_nodes: Optional[int] = None,
) -> Union[Tuple[Tensor, OptTensor], SparseTensor]:

    sparsetensor_flag = isinstance(edge_index, SparseTensor)

    if sparsetensor_flag:
        row, col, value = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = value
    else:
        value = edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if value is not None:
        if fill_value is None:
            loop_attr = value.new_full((N,) + value.size()[1:], 1.0)
        elif isinstance(fill_value, (int, float)):
            loop_attr = value.new_full((N,) + value.size()[1:], fill_value)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(value.device, value.dtype)
            if value.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)
        elif isinstance(fill_value, str):
            loop_attr = scatter(value, edge_index[1], dim=0, dim_size=N, reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        inv_mask = ~mask
        loop_attr[edge_index[0][inv_mask]] = value[inv_mask]

        edge_attr = torch.cat([value[mask], loop_attr], dim=0)
    else:
        edge_attr = None

    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    if sparsetensor_flag:
        return SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_attr,
            sparse_sizes=(N, N),
        )
    else:
        return edge_index, edge_attr


def gcn_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    improved=False,
    add_self_loops=True,
    dtype=None,
    gra_part=False,
    only_train_data=False,
    args=None,
):

    fill_value = 2.0 if improved else 1.0
    if (
        gra_part is True and edge_weight is not None and not isinstance(edge_index, SparseTensor)
    ) or args.flowgnn_debug is True:
        return edge_index, edge_weight

    elif (gra_part is True and isinstance(edge_index, SparseTensor)) or args.flowgnn_debug is True:
        return edge_index, edge_weight

    elif isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        adj_t = adj_t.fill_value(1.0, dtype=dtype)
        adj_t = add_remaining_self_loops_sparse(adj_t, fill_value=fill_value, num_nodes=num_nodes)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

        if edge_weight is None:
            edge_weight = torch.ones((adj_t.nnz(),), dtype=dtype, device=adj_t.device())

        edge_weight = (
            deg_inv_sqrt[adj_t.storage.row()] * edge_weight * deg_inv_sqrt[adj_t.storage.col()]
        )
        edge_index = adj_t.set_value(edge_weight, layout="coo")

        return edge_index, edge_weight

    elif only_train_data is True:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes
            )
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

        # edge_weight = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes
            )
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

        # edge_weight = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    # _cached_adj_t: Optional[SparseTensor]
    _cached_adj_t = None
    _cached_edge_index = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        args=None,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        layer: int = None,
        **kwargs,
    ):

        kwargs.setdefault("aggr", "add")
        super(GCNConv, self).__init__(**kwargs)
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.layer = layer

        self._cached_edge_index = None
        # self._cached_adj_t = None
        self.no_edge_weight = args.no_edge_weight
        self.original_edge_weight = args.original_edge_weight

        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                  weight_initializer='glorot')
        if (
            (layer != (args.num_layers - 1) or not args.unstructured_for_last_layer)
            and args.nmsparsity
            and args.enable_mask
        ):
            self.lin = NMSparseMultiLinear(in_channels, out_channels, args=args, layer=layer)
        elif (
            layer != (args.num_layers - 1) or not args.unstructured_for_last_layer
        ) and args.nmsparsity:
            self.lin = NMSparseLinear(in_channels, out_channels, args=args, layer=layer)
        elif args.enable_mask is True:
            self.lin = SparseLinearMulti_mask(in_channels, out_channels, args=args, layer=layer)
        else:
            self.lin = SparseLinear(in_channels, out_channels, args=args, layer=layer)

        # if bias:
        #    self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #    self.register_parameter('bias', None)

        # self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        # self._cached_edge_index = None
        # self._cached_adj_t = None

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

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        threshold,
        sparsity=None,
        edge_weight: OptTensor = None,
        args=None,
        epoch=None,
        part=None,
    ) -> Tensor:
        if self.normalize:
            if self.args.gra_part or self.args.only_train_data or self.args.flowgnn_debug:
                if self.args.flowgnn_debug:
                    # NO SELF LOOP
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        False,  # add self loop = False
                        gra_part=self.args.gra_part,
                        args=self.args,
                    )
                else:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        gra_part=self.args.gra_part,
                        args=self.args,
                    )
            else:
                if isinstance(edge_index, Tensor):
                    cache = GCNConv._cached_edge_index
                    # cache = self._cached_edge_index
                    if cache is None:
                        edge_index, edge_weight = gcn_norm(
                            edge_index,
                            edge_weight,
                            x.size(self.node_dim),
                            self.improved,
                            self.add_self_loops,
                            args=self.args,
                        )
                        if self.cached:
                            GCNConv._cached_edge_index = (
                                edge_index,
                                edge_weight,
                            )
                    else:
                        edge_index, edge_weight = cache
                elif isinstance(edge_index, SparseTensor):
                    cache = GCNConv._cached_adj_t
                    if cache is None:
                        edge_index, edge_weight = gcn_norm(
                            edge_index,
                            edge_weight,
                            x.size(self.node_dim),
                            self.improved,
                            self.add_self_loops,
                            args=self.args,
                        )
                        if self.cached:
                            GCNConv._cached_adj_t = (edge_index, edge_weight)
                    else:
                        edge_index, edge_weight = cache

        if (
            args.nmsparsity
            and self.layer == args.num_layers - 1
            and args.unstructured_for_last_layer
        ):

            threshold = self.get_threshold(sparsity, epoch=epoch)
        # np.savetxt("edge_weight.txt", edge_weight.detach().cpu())
        x = self.lin(x, threshold, sparsity)

        # if args.sparsity_profiling:
        #     num_zeros = (x == 0).sum().item()
        #     num_elements = x.numel()
        #     sparsity = num_zeros / num_elements
        #     print(f"XW sparsity: {sparsity:.8f}")

        if args.flowgnn_debug:

            os.makedirs(
                f"./pretrained_model/Output/l{self.layer+1}",
                exist_ok=True,
            )
            # Save logits to txt file
            # if args.outgoing_kmeans:
            # xw = x.detach().cpu().numpy()
            # logits_flattened = xw.flatten()
            # np.savetxt(
            #     f"./pretrained_model/Output/l{self.layer}/l{self.layer}_part{part}_xw_for_partition.txt",
            #     logits_flattened,
            # )

            with open(
                f"./pretrained_model/Output/l{self.layer+1}/l{self.layer+1}_part{part}_XW.txt",
                "w",
            ) as file:
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        file.write("%f\n" % x[i][j])

        # if self.args.validate:
        #     num_zeros = (x == 0).sum().item()
        #     num_elements = x.numel()
        #     sparsity = num_zeros / num_elements
        #     print(f"XW sparsity: {sparsity:.8f}")

        if self.args.flowgnn_debug:
            out = self.propagate(edge_index, x=x, edge_weight=None, size=None)  # edge_weight = None
        else:
            out = self.propagate(
                edge_index,
                x=x,
                edge_weight=edge_weight,
                size=None,
            )

        if self.args.flowgnn_debug:
            num_zeros = (out == 0).sum().item()
            num_elements = out.numel()
            sparsity = num_zeros / num_elements
            # if self.args.flowgnn_debug:
            print(f"AXW sparsity: {sparsity:.8f}")
            os.makedirs(f"./pretrained_model/Output/l{self.layer+1}", exist_ok=True)

            with open(
                f"./pretrained_model/Output/l{self.layer+1}/l{self.layer+1}_AXW_whole_graph.txt",
                "w",
            ) as file:
                for i in range(out.shape[0]):
                    for j in range(out.shape[1]):
                        file.write("%f\n" % out[i][j])

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if self.no_edge_weight is True or edge_weight is None:
            return x_j
        else:
            # edge_weight = None
            return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor, edge_weight: OptTensor) -> Tensor:
    #     if edge_weight is None:
    #         return matmul(adj_t, x, reduce=self.aggr)
    #     else:
    #         x if edge_weight is None else edge_weight.view(-1, 1) * x
    #         return matmul(adj_t, x, reduce=self.aggr)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)

    def rerandomize(self, mode, la, mu):
        self.lin.rerandomize(mode, la, mu)


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        args=None,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        layer: int = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.args = args
        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            # self.lin_src = Linear(in_channels, heads * out_channels,
            #                      bias=False, weight_initializer='glorot')

            if (
                (layer != (args.num_layers - 1) or not args.unstructured_for_last_layer)
                and args.nmsparsity
                and args.enable_mask
            ):
                self.lin_src = NMSparseMultiLinear(
                    in_channels, heads * out_channels, args=args, layer=layer
                )
            elif (
                layer != (args.num_layers - 1) or not args.unstructured_for_last_layer
            ) and args.nmsparsity:
                self.lin_src = NMSparseLinear(
                    in_channels, heads * out_channels, args=args, layer=layer
                )
            elif args.enable_mask is True:
                self.lin_src = SparseLinearMulti_mask(
                    in_channels, heads * out_channels, args=args, layer=layer
                )
            else:
                self.lin_src = SparseLinear(
                    in_channels, heads * out_channels, args=args, layer=layer
                )
            self.lin_dst = self.lin_src

            # if args.enable_mask is True:
            #     self.lin_src = SparseLinearMulti_mask(
            #         in_channels, heads * out_channels, args=args, layer=layer
            #     )
            #     self.lin_dst = self.lin_src

            # else:
            #     self.lin_src = SparseLinear(
            #         in_channels, heads * out_channels, args=args, layer=layer
            #     )
            #     self.lin_dst = self.lin_src

        else:
            # self.lin_src = Linear(in_channels[0], heads * out_channels, False,
            #                      weight_initializer='glorot')
            # self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
            #                      weight_initializer='glorot')
            if args.enable_mask is True:
                self.lin_src = SparseLinearMulti_mask(
                    in_channels[0],
                    heads * out_channels,
                    args=args,
                    layer=layer,
                )
                self.lin_dst = SparseLinearMulti_mask(
                    in_channels[1],
                    heads * out_channels,
                    args=args,
                    layer=layer,
                )

            else:
                self.lin_src = SparseLinear(
                    in_channels[0],
                    heads * out_channels,
                    args=args,
                    layer=layer,
                )
                self.lin_dst = SparseLinear(
                    in_channels[1],
                    heads * out_channels,
                    args=args,
                    layer=layer,
                )

        # The learnable parameters to compute attention coefficients:
        # self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        # self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        if args.enable_mask is True:
            self.att_src = SparseParameterMulti_mask(heads, out_channels, args=args, layer=layer)
            self.att_dst = SparseParameterMulti_mask(heads, out_channels, args=args, layer=layer)
        else:
            self.att_src = SparseParameter(heads, out_channels, args=args, layer=layer)
            self.att_dst = SparseParameter(heads, out_channels, args=args, layer=layer)

        if edge_dim is not None:
            # self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
            #                       weight_initializer='glorot')
            if args.enable_mask is True:
                self.lin_edge = SparseLinearMulti_mask(
                    edge_dim, heads * out_channels, args=args, layer=layer
                )
                # self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
                self.att_edge = SparseParameterMulti_mask(
                    heads, out_channels, args=args, layer=layer
                )

            else:
                self.lin_edge = SparseLinear(edge_dim, heads * out_channels, args=args, layer=layer)
                # self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
                self.att_edge = SparseParameter(heads, out_channels, args=args, layer=layer)

        else:
            self.lin_edge = None
            self.register_parameter("att_edge", None)

        self._alpha = None

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

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

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        threshold,
        sparsity=None,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
        epoch=None,
    ):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        if (
            self.args.nmsparsity
            and self.layer == self.args.num_layers - 1
            and self.args.unstructured_for_last_layer
        ):

            threshold = self.get_threshold(sparsity, epoch=epoch)

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x, threshold, sparsity).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src, threshold, sparsity).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst, threshold, sparsity).view(-1, H, C)

        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src(threshold)).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst(threshold)).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                # print("edge_index shape",edge_index.shape)
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #   out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        threshold = None

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr, threshold)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge(threshold)).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def rerandomize(self, mode, la, mu, threshold):
        self.lin_src.rerandomize(mode, la, mu, threshold)
        self.lin_dst.rerandomize(mode, la, mu, threshold)
        self.att_src.rerandomize(mode, la, mu, threshold)
        self.att_dst.rerandomize(mode, la, mu, threshold)
        if self.edge_dim is not None:
            self.lin_edge.rerandomize(mode, la, mu, threshold)
            self.att_edge.rerandomize(mode, la, mu, threshold)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )


class dmGATConv(MessagePassing):
    r"""https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/dense_gat_conv.html

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        args=None,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            # self.lin_src = Linear(in_channels, heads * out_channels,
            #                      bias=False, weight_initializer='glorot')
            if args.enable_mask is True:
                self.lin_src = SparseLinearMulti_mask(in_channels, heads * out_channels, args=args)
                self.lin_dst = self.lin_src

            else:
                self.lin_src = SparseLinear(in_channels, heads * out_channels, args=args)
                self.lin_dst = self.lin_src

        else:
            # self.lin_src = Linear(in_channels[0], heads * out_channels, False,
            #                      weight_initializer='glorot')
            # self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
            #                      weight_initializer='glorot')
            if args.enable_mask is True:
                self.lin_src = SparseLinearMulti_mask(
                    in_channels[0], heads * out_channels, args=args
                )
                self.lin_dst = SparseLinearMulti_mask(
                    in_channels[1], heads * out_channels, args=args
                )

            else:
                self.lin_src = SparseLinear(in_channels[0], heads * out_channels, args=args)
                self.lin_dst = SparseLinear(in_channels[1], heads * out_channels, args=args)

        # The learnable parameters to compute attention coefficients:
        # self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        # self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        if args.enable_mask is True:
            self.att_src = SparseParameterMulti_mask(heads, out_channels, args=args)
            self.att_dst = SparseParameterMulti_mask(heads, out_channels, args=args)
        else:
            self.att_src = SparseParameter(heads, out_channels, args=args)
            self.att_dst = SparseParameter(heads, out_channels, args=args)

        if edge_dim is not None:
            # self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
            #                       weight_initializer='glorot')
            if args.enable_mask is True:
                self.lin_edge = SparseLinearMulti_mask(edge_dim, heads * out_channels, args=args)
                # self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
                self.att_edge = SparseParameterMulti_mask(heads, out_channels, args=args)

            else:
                self.lin_edge = SparseLinear(edge_dim, heads * out_channels, args=args)
                # self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
                self.att_edge = SparseParameter(heads, out_channels, args=args)

        else:
            self.lin_edge = None
            self.register_parameter("att_edge", None)

        self._alpha = None

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        adj,
        threshold,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
    ):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]
        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        # add loop
        adj = adj.clone()
        idx = torch.arange(N, dtype=torch.long, device=adj.device)
        adj[:, idx, idx] = 1.0

        x = self.lin_src(x, threshold).view(B, N, H, C)  # [B, N, H, C]  first combination XW

        # x = (x_src, x_dst)

        alpha_src = (x * self.att_src(threshold)).sum(dim=-1)
        alpha_dst = None if x is None else (x * self.att_dst(threshold)).sum(-1)
        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)  # [B, N, N, H] calculate alpha

        # Weighted and masked softmax:
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1))  # aggregation A(XW)
        out = out.movedim(1, 2)  # [B,N,H,C]
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa

        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        # if self.bias is not None:
        #   out += self.bias
        out = out.squeeze(0)
        return out

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        threshold = None

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr, threshold)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge(threshold)).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def rerandomize(self, mode, la, mu, threshold):
        self.lin_src.rerandomize(mode, la, mu, threshold)
        self.lin_dst.rerandomize(mode, la, mu, threshold)
        self.att_src.rerandomize(mode, la, mu, threshold)
        self.att_dst.rerandomize(mode, la, mu, threshold)
        if self.edge_dim is not None:
            self.lin_edge.rerandomize(mode, la, mu, threshold)
            self.att_edge.rerandomize(mode, la, mu, threshold)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )


class Graphlevel_GCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        args=None,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = False,
        bias: bool = True,
        layer: int = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(Graphlevel_GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.args = args
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.linear_sparsity = args.linear_sparsity
        self.enable_abs_comp = args.enable_abs_comp
        self.SLTRoot = args.SLTRoot
        self.train_mode = args.train_mode
        self.init_mode_mask = args.init_mode_mask
        self.init_scale_score = args.init_scale_score
        self.SLTRoot_ini = args.SLTRoot_ini
        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                  weight_initializer='glorot')
        self.layer = layer

        if args.train_mode == "score_only":

            if args.nmsparsity and args.enable_mask:
                self.lin = NMSparseMultiLinear(in_channels, out_channels, args=args, layer=layer)
            elif args.nmsparsity:
                self.lin = NMSparseLinear(in_channels, out_channels, args=args, layer=layer)
            elif args.enable_mask is True:
                self.lin = SparseLinearMulti_mask(in_channels, out_channels, args=args, layer=layer)
            else:
                self.lin = SparseLinear(in_channels, out_channels, args=args, layer=layer)

            # if args.enable_mask is True:
            #     if args.adp is True:
            #         self.lin = SparseLinearMulti_mask_adp(
            #             in_channels, out_channels, args=args, layer=layer
            #         )
            #     else:
            #         self.lin = SparseLinearMulti_mask(
            #             in_channels, out_channels, args=args, layer=layer
            #         )
            # else:
            #     self.lin = SparseLinear(in_channels, out_channels, args=args, layer=layer)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)

        if args.dataset == "hep10k":
            pass
        elif args.train_mode == "score_only" and args.SLTBond is True:
            self.bond_encoder = SLT_BondEncoder(emb_dim=args.dim_hidden, args=args, layer=layer)
        # elif args.train_mode == "score_only" and args.SLTBond is False:
        #     self.bond_encoder = LT_BondEncoder(emb_dim=args.dim_hidden)
        else:
            self.bond_encoder = BondEncoder(emb_dim=args.dim_hidden)

        # self.root_emb = torch.tensor()
        # self.root_emb = nn.init.xavier_uniform_(1, args.dim_hidden)

        if args.dataset == "hep10k":
            pass
        elif args.train_mode == "score_only" and args.SLTRoot is True:
            self.root_emb = torch.nn.Embedding(1, args.dim_hidden)
            self.root_emb.weight.requires_grad = False
            self.root_emb_weight_score = nn.Parameter(torch.ones(1, self.root_emb.weight.size(1)))
            self.root_emb_weight_score.is_score = True
            self.root_emb_weight_score.root = True
            self.root_emb_weight_score.sparsity = self.linear_sparsity
            SparseModule.init_param_(
                self,
                self.root_emb_weight_score,
                init_mode=self.init_mode_mask,
                scale=self.init_scale_score,
            )
            # print(self.root_emb_weight_score.type)
            self.root_emb_weight_ones = torch.ones(1, self.root_emb.weight.size(1))
            self.root_emb_weight_zeros = torch.zeros(1, self.root_emb.weight.size(1))
            self.root_emb_weight_ones.requires_grad = False
            self.root_emb_weight_zeros.requires_grad = False
            if self.SLTRoot_ini == "default":
                pass
            else:
                SparseModule.init_param_(
                    self,
                    self.root_emb.weight,
                    init_mode=self.SLTRoot_ini,
                    scale=self.init_scale_score,
                    sparse_value=self.linear_sparsity,
                    layer=layer,
                )
            self.root_emb.weight.requires_grad = False

        # elif args.train_mode == "score_only" and args.SLTRoot is False:
        #     self.root_emb = torch.nn.Embedding(1, args.dim_hidden)
        #     self.root_emb = resetEMB_is_score(self.root_emb)
        #     if self.SLTRoot_ini == "default":
        #         pass
        #     else:
        #         init_param_(
        #             self.root_emb.weight,
        #             init_mode=self.SLTRoot_ini,
        #             scale=self.init_scale_score,
        #             sparse_value=self.linear_sparsity,
        #         )

        else:
            self.root_emb = torch.nn.Embedding(1, args.dim_hidden)
            if self.SLTRoot_ini == "default":
                pass
            else:
                SparseModule.init_param_(
                    self,
                    self.root_emb.weight,
                    init_mode=self.SLTRoot_ini,
                    scale=self.init_scale_score,
                    sparse_value=self.linear_sparsity,
                )

        self.enable_multi_mask = args.enable_mask

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def percentile(self, t, q):
        t_flat = t.view(-1)
        t_sorted, _ = torch.sort(t_flat)
        k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
        return t_sorted[k].item()

    def bond_get_threshold(self, sparsity, layer=None):
        if self.args.local_pruning is True:
            # 外側のリストの初期化
            outer_threshold_list = []

            # 各bond属性を持つ重みに対するループ (0から2まで)
            for bond_idx in range(3):
                bond_str = f".{bond_idx}"

                # 対象のbond属性を持つ重みだけをフィルタリング
                filtered_params = []
                for name, p in self.named_parameters():
                    # print(name)
                    if (
                        bond_str in name
                        and hasattr(p, "is_score")
                        and hasattr(p, "bond")
                        and p.is_score
                        and p.sparsity == self.linear_sparsity
                    ):
                        # # 追加: 条件を満たすパラメータの名前を表示
                        # print(name)

                        filtered_params.append(p)

                if self.enable_multi_mask:
                    threshold_list = []
                    for value in sparsity:
                        local = [p.detach().flatten() for p in filtered_params]
                        local = torch.cat(local)
                        if self.enable_abs_comp is False:
                            threshold = self.percentile(local, value * 100)
                        else:
                            threshold = self.percentile(local.abs(), value * 100)
                        threshold_list.append(threshold)
                    outer_threshold_list.append(threshold_list)
                else:
                    local = [p.detach().flatten() for p in filtered_params]
                    local = torch.cat(local)
                    if self.enable_abs_comp is False:
                        threshold = self.percentile(local, sparsity * 100)
                    else:
                        threshold = self.percentile(local.abs(), sparsity * 100)
                    outer_threshold_list.append([threshold])
                    # print(name)
            return outer_threshold_list

        elif self.args.local_pruning is False:
            if self.enable_multi_mask is True:  # enable multi-mask
                threshold_list = []
                for value in sparsity:
                    local = []
                    for name, p in self.named_parameters():
                        if (
                            hasattr(p, "is_score")
                            and hasattr(p, "bond")
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
                        and hasattr(p, "bond")
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

    def root_get_threshold(self, sparsity, layer=None):
        if self.args.local_pruning is True:
            if self.enable_multi_mask is True:  # enable multi-mask
                threshold_list = []
                for value in sparsity:
                    local = []
                    for name, p in self.named_parameters():
                        if (
                            hasattr(p, "is_score")
                            and hasattr(p, "root")
                            and p.is_score
                            and p.sparsity == self.linear_sparsity
                            # and f".{layer}." in name
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
                    # print(name)
                    # print(self._get_name)
                    if (
                        hasattr(p, "is_score")
                        and hasattr(p, "root")
                        and p.is_score
                        and p.sparsity == self.linear_sparsity
                        # and f"{layer}" in self._get_name()
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

        elif self.args.local_pruning is False:
            if self.enable_multi_mask is True:  # enable multi-mask
                threshold_list = []
                for value in sparsity:
                    local = []
                    for name, p in self.named_parameters():
                        if (
                            hasattr(p, "is_score")
                            and hasattr(p, "root")
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
                        and hasattr(p, "root")
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

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr,
        threshold=None,
        index_mask=None,
        sparsity=None,
        layer=None,
        batched_data=None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        if self.args.train_mode == "score_only":
            if self.args.flowgnn_debug:
                # Calculate sparsity for each data point in the encoded batch
                encoded_sparsity = []
                for i in range(batched_data.num_graphs):
                    mask = batched_data.batch == i
                    h_i = x[mask]  # Get the encoded nodes for the i-th graph
                    num_elements_i = h_i.numel()
                    non_zero_elements_i = (h_i != 0).sum().item()
                    zero_elements_i = num_elements_i - non_zero_elements_i
                    zero_ratio_i = zero_elements_i / num_elements_i
                    encoded_sparsity.append(zero_ratio_i)

                os.makedirs(
                    f"./pretrained_model/Output/L{self.layer+1}",
                    exist_ok=True,
                )
                before_XW_x = x.detach().cpu().numpy()
                before_XW_x = before_XW_x.T
                with open(
                    f"./pretrained_model/Output/L{self.layer+1}/before_XW_x.bin",
                    "wb",
                ) as f:
                    f.write(before_XW_x.tobytes())
                before_XW_x = before_XW_x.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Output/L{self.layer+1}/before_XW_x.txt",
                    before_XW_x,
                    fmt="%.6f",
                    delimiter=",",
                )
            x = self.lin(x, threshold, sparsity)
            if self.args.flowgnn_debug:
                os.makedirs(
                    f"./pretrained_model/Output/L{self.layer+1}",
                    exist_ok=True,
                )
                after_XW_x = x.detach().cpu().numpy()
                after_XW_x = after_XW_x.T
                with open(
                    f"./pretrained_model/Output/L{self.layer+1}/after_XW_x.bin",
                    "wb",
                ) as f:
                    f.write(after_XW_x.tobytes())
                after_XW_x = after_XW_x.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Output/L{self.layer+1}/after_XW_x.txt",
                    after_XW_x,
                    fmt="%.6f",
                    delimiter=",",
                )
                # Calculate sparsity for each data point in the encoded batch
                encoded_sparsity = []
                for i in range(batched_data.num_graphs):
                    mask = batched_data.batch == i
                    h_i = x[mask]  # Get the encoded nodes for the i-th graph
                    num_elements_i = h_i.numel()
                    non_zero_elements_i = (h_i != 0).sum().item()
                    zero_elements_i = num_elements_i - non_zero_elements_i
                    zero_ratio_i = zero_elements_i / num_elements_i
                    encoded_sparsity.append(zero_ratio_i)

                # print(f"XW sparsity: {encoded_sparsity}")
        else:
            # if self.args.validate:
            #     # Calculate non-zero, zeros and zero ratio for x
            #     x_total_elements = torch.numel(x)
            #     x_zero_elements = torch.sum(x == 0).item()
            #     x_non_zero_elements = x_total_elements - x_zero_elements
            #     x_zero_ratio = x_zero_elements / x_total_elements

            #     print(
            #         f"X: zero:{x_zero_elements}, non_zero:{x_non_zero_elements}, ",
            #         f"zero_ratio:{x_zero_ratio}, shape:{x.shape}",
            #     )

            x = self.lin(x)
            # 重みとバイアスを取得
            if self.args.flowgnn_debug:
                os.makedirs(
                    f"./pretrained_model/Models/WandB/L{self.layer+1}",
                    exist_ok=True,
                )
                weight = self.lin.weight.detach().cpu().numpy()
                weight = weight.T
                bias = self.lin.bias.detach().cpu().numpy()
                bias = bias.T

                # 重みをバイナリファイルとして保存
                with open(
                    f"./pretrained_model/Models/WandB/L{self.layer+1}/weight_{weight.shape}.bin",
                    "wb",
                ) as f:
                    f.write(weight.tobytes())

                # 重みをテキストファイルとして保存
                weight_flat = weight.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/WandB/L{self.layer+1}/weight_{weight.shape}.txt",
                    weight_flat,
                    delimiter=",",
                    fmt="%.6f",
                )

                # バイアスをバイナリファイルとして保存
                with open(
                    f"./pretrained_model/Models/WandB/L{self.layer+1}/bias_{bias.shape}.bin",
                    "wb",
                ) as f:
                    f.write(bias.tobytes())

                # バイアスをテキストファイルとして保存
                bias_flat = bias.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/WandB/L{self.layer+1}/bias_{bias.shape}.txt",
                    bias_flat,
                    delimiter=",",
                    fmt="%.6f",
                )
        if self.args.dataset == "hep10k":
            pass
        elif (
            self.args.train_mode == "score_only"
            and self.args.SLTBond is True
            and self.args.local_pruning is True
        ):
            bond_sparsity = sparsity
            bond_threshold = self.bond_get_threshold(bond_sparsity, layer=layer)
            edge_embedding = self.bond_encoder(edge_attr, bond_threshold)
            # add for root_emb_SLT
            root_threshold = self.root_get_threshold(bond_sparsity, layer=layer)
        elif (
            self.args.train_mode == "score_only"
            and self.args.SLTBond is True
            and self.args.local_pruning is False
        ):
            bond_threshold = self.bond_get_threshold(sparsity)
            if self.args.flowgnn_debug:
                os.makedirs(
                    f"./pretrained_model/Output/L{self.layer+1}",
                    exist_ok=True,
                )
                before_bond_edge_attr = edge_attr.detach().cpu().numpy()
                before_bond_edge_attr = before_bond_edge_attr.T
                with open(
                    f"./pretrained_model/Output/L{self.layer+1}/before_bond_edge_attr.bin",
                    "wb",
                ) as f:
                    f.write(before_bond_edge_attr.tobytes())
                before_bond_edge_attr = before_bond_edge_attr.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Output/L{self.layer+1}/before_bond_edge_attr.txt",
                    before_bond_edge_attr,
                    fmt="%d",
                    delimiter=",",
                )
            edge_embedding = self.bond_encoder(edge_attr, bond_threshold)
            if self.args.flowgnn_debug:
                os.makedirs(
                    f"./pretrained_model/Output/L{self.layer+1}",
                    exist_ok=True,
                )
                after_bond_edge_attr = edge_embedding.detach().cpu().numpy()
                after_bond_edge_attr = after_bond_edge_attr.T
                with open(
                    f"./pretrained_model/Output/L{self.layer+1}/after_bond_edge_attr.bin",
                    "wb",
                ) as f:
                    f.write(after_bond_edge_attr.tobytes())
                after_bond_edge_attr = after_bond_edge_attr.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Output/L{self.layer+1}/after_bond_edge_attr.txt",
                    after_bond_edge_attr,
                    fmt="%.6f",
                    delimiter=",",
                )
        else:
            edge_embedding = self.bond_encoder(edge_attr)
            if self.args.flowgnn_debug:
                all_bond_embs = []

                for i in range(3):
                    print_bond_emb = (
                        getattr(self.bond_encoder.bond_embedding_list, str(i))
                        .weight.detach()
                        .cpu()
                        .numpy()
                    )
                    print_bond_emb = print_bond_emb.T
                    all_bond_embs.append(print_bond_emb)

                # 結合して保存
                all_bond_embs = np.concatenate(all_bond_embs, axis=1)
                all_bond_embs = all_bond_embs.T
                shape = all_bond_embs.shape

                os.makedirs(
                    f"./pretrained_model/Models/Bond/L{self.layer+1}",
                    exist_ok=True,
                )

                with open(
                    f"./pretrained_model/Models/Bond/L{self.layer+1}/bond_emb_combined_{shape[0]}x{shape[1]}.bin",
                    "wb",
                ) as f:
                    f.write(all_bond_embs.tobytes())

                all_bond_embs = all_bond_embs.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/Bond/L{self.layer+1}/bond_emb_combined_{shape[0]}x{shape[1]}.txt",
                    all_bond_embs,
                    fmt="%.6f",
                    delimiter=",",
                )

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if self.args.dataset == "hep10k":
            propagete = self.propagate(edge_index, x=x, edge_attr=0, norm=norm)
            GCNConv_output = propagete + F.relu(x) * 1.0 / deg.view(-1, 1)
        elif self.train_mode == "score_only" and self.SLTRoot is True:
            root_threshold = self.root_get_threshold(bond_sparsity, layer=layer)
            if isinstance(root_threshold, float):
                root_threshold = [root_threshold]
            root_embsubnets = []
            for threshold_v in root_threshold:
                if self.enable_abs_comp is False:
                    root_embsubnet = GetSubnet.apply(
                        self.root_emb_weight_score,
                        threshold_v,
                        self.root_emb_weight_zeros,
                        self.root_emb_weight_ones,
                    )
                else:
                    root_embsubnet = GetSubnet.apply(
                        self.root_emb_weight_score.abs(),
                        threshold_v,
                        self.root_emb_weight_zeros,
                        self.root_emb_weight_ones,
                    )
                root_embsubnets.append(root_embsubnet)
            rootcombined_subnet = torch.stack(root_embsubnets).sum(dim=0)
            root_emb_tensor = self.root_emb.weight
            modified_root_emb = (
                root_emb_tensor * rootcombined_subnet
            )  # Applying the subnet operation on the tensor
            # if self.args.validate and self.args.flowgnn_debug is True:
            #     rootcombined_subnet = (
            #         rootcombined_subnet.detach().cpu().numpy()
            #     )
            #     with open(
            #         f"./layer{self.layer}_Bond_rootcombined_subnet_{rootcombined_subnet.shape}.bin",
            #         "wb",
            #     ) as f:
            #         f.write(rootcombined_subnet.tobytes())
            #     np.savetxt(
            #         f"./layer{self.layer}_Bond_rootcombined_subnet{rootcombined_subnet.shape}.txt",
            #         rootcombined_subnet,
            #         delimiter=",",
            #     )

            #     root_emb_tensor = root_emb_tensor.detach().cpu().numpy()
            #     # 結果を保存するための1次元配列を初期化
            #     result = np.zeros(
            #         root_emb_tensor.shape[0] * root_emb_tensor.shape[1],
            #         dtype=int,
            #     )

            #     # root_emb_tensorの値を1または0に変換して結果の配列に格納
            #     index = 0
            #     for i in range(root_emb_tensor.shape[1]):
            #         for j in range(root_emb_tensor.shape[0]):
            #             if root_emb_tensor[j, i] > 0:
            #                 result[index] = 1
            #             else:
            #                 result[index] = 0
            #             index += 1

            #     # 結果をテキストファイルに保存
            #     np.savetxt(
            #         f"./layer{self.layer}_Bond_binary_root_emb_tensor{root_emb_tensor.shape}.txt",
            #         result,
            #         fmt="%d",
            #     )

            #     # Save as binary file
            #     with open(
            #         f"./layer{self.layer}_Bond_root_emb_tensor{root_emb_tensor.shape}.bin",
            #         "wb",
            #     ) as f:
            #         f.write(root_emb_tensor.tobytes())

            #     # Save as text file
            #     np.savetxt(
            #         f"./layer{self.layer}_Bond_root_emb_tensor{root_emb_tensor.shape}.txt",
            #         root_emb_tensor,
            #         delimiter=",",
            #     )

            GCNConv_output = self.propagate(
                edge_index, x=x, edge_attr=edge_embedding, norm=norm
            ) + F.relu(x + modified_root_emb) * 1.0 / deg.view(-1, 1)
        else:
            # root_emb_tensor = self.root_emb(torch.tensor([0]).to(device))  # Extracting the embedding tensor
            if self.args.flowgnn_debug:
                os.makedirs(
                    f"./pretrained_model/Output/L{self.layer+1}",
                    exist_ok=True,
                )
                before_AXW_x = x.detach().cpu().numpy()
                before_AXW_x = before_AXW_x.T
                with open(
                    f"./pretrained_model/Output/L{self.layer+1}/before_AXW_x.bin",
                    "wb",
                ) as f:
                    f.write(before_AXW_x.tobytes())
                before_AXW_x = before_AXW_x.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Output/L{self.layer+1}/before_AXW_x.txt",
                    before_AXW_x,
                    fmt="%.6f",
                    delimiter=",",
                )

            propagete = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
            GCNConv_output = propagete + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)
            if self.args.flowgnn_debug:
                os.makedirs(
                    f"./pretrained_model/Output/L{self.layer+1}",
                    exist_ok=True,
                )
                after_propagate = propagete.detach().cpu().numpy()
                after_propagate = after_propagate.T

                with open(
                    f"./pretrained_model/Output/L{self.layer+1}/after_propagate.bin",
                    "wb",
                ) as f:
                    f.write(after_propagate.tobytes())
                after_propagate = after_propagate.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Output/L{self.layer+1}/after_propagate.txt",
                    after_propagate,
                    fmt="%.6f",
                    delimiter=",",
                )
                os.makedirs(
                    f"./pretrained_model/Output/L{self.layer+1}",
                    exist_ok=True,
                )
                after_AXW_x = GCNConv_output.detach().cpu().numpy()
                after_AXW_x = after_AXW_x.T

                with open(
                    f"./pretrained_model/Output/L{self.layer+1}/after_AXW_x.bin",
                    "wb",
                ) as f:
                    f.write(after_AXW_x.tobytes())
                after_AXW_x = after_AXW_x.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Output/L{self.layer+1}/after_AXW_x.txt",
                    after_AXW_x,
                    fmt="%.6f",
                    delimiter=",",
                )
                os.makedirs(
                    f"./pretrained_model/Models/Root/L{self.layer+1}",
                    exist_ok=True,
                )
                print_root_emb = self.root_emb.weight.detach().cpu().numpy()
                print_root_emb = print_root_emb.T
                with open(
                    f"./pretrained_model/Models/Root/L{self.layer+1}/root_emb.bin",
                    "wb",
                ) as f:
                    f.write(print_root_emb.tobytes())
                print_root_emb = print_root_emb.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Models/Root/L{self.layer+1}/root_emb.txt",
                    print_root_emb,
                    fmt="%.6f",
                    delimiter=",",
                )

        return GCNConv_output

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
