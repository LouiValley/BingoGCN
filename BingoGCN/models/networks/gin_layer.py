# import pdb

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_modules import (
    NMSparseLinear,
    NMSparseMultiLinear,
    SparseLinear,
    SparseLinearMulti_mask,
)

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""
msg_mask = fn.src_mul_edge("h", "mask", "m")
# msg_mask = fn.u_mul_e('h', 'mask', 'm')
msg_orig = fn.copy_u("h", "m")


def resetBN_is_score(tmpBN):
    tmpBN.weight.is_score = True
    tmpBN.bias.is_score = True
    tmpBN.weight.sparsity = 0.0
    tmpBN.bias.sparsity = 0.0
    return tmpBN


class GINLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    graph_norm :
        boolean flag for output features normalization w.r.t. graph sizes.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.

    """

    def __init__(
        self,
        apply_func,
        aggr_type,
        dropout,
        graph_norm,
        batch_norm,
        residual=False,
        init_eps=0,
        learn_eps=False,
        args=None,
        layer=None,
    ):
        super().__init__()
        self.apply_func = apply_func
        self.args = args
        if aggr_type == "sum":
            self._reducer = fn.sum
        elif aggr_type == "max":
            self._reducer = fn.max
        elif aggr_type == "mean":
            self._reducer = fn.mean
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggr_type))

        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim
        self.BN_GIN = args.BN_GIN
        self.train_mode = args.train_mode

        if in_dim != out_dim:
            self.residual = False

        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

        if self.BN_GIN and self.train_mode == "score_only":
            self.bn_node_h = resetBN_is_score(nn.BatchNorm1d(out_dim))
        else:
            self.bn_node_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, snorm_n, threshold, sparsity=None):

        # h_in = h  # for residual connection
        device = g.device
        h = h.to(device)
        g = g.local_var()
        g.ndata["h"] = h
        # g.update_all(msg_orig, self._reducer('m', 'neigh'))
        # pruning edges by cutting message passing process
        g.update_all(msg_mask, self._reducer("m", "neigh"))

        h = (1 + self.eps) * h + g.ndata["neigh"]
        if self.apply_func is not None:
            h = self.apply_func(h, threshold, sparsity)

        # if self.graph_norm:
        #    h = h * snorm_n # normalize activation w.r.t. graph size

        # if self.batch_norm:
        #    h = self.bn_node_h(h) # batch normalization
        # if self.BN_GIN and self.train_mode == "score_only":
        #     h = self.bn_node_h(h)  # batch normalization

        h = F.relu(h)  # non-linear activation

        # if self.residual:
        #    h = h_in + h # residual connection

        h = F.dropout(h, self.dropout, training=self.training)

        return h


class ApplyNodeFunc(nn.Module):
    """
    This class is used in class GINNet
    Update the node feature hv with MLP
    """

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h, threshold, sparsity):
        h = self.mlp(h, threshold, sparsity)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, args, layer=None):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            if args.nmsparsity and args.enable_mask:
                self.linear = NMSparseMultiLinear(input_dim, output_dim, args=args, layer=layer)
            elif args.nmsparsity:
                self.linear = NMSparseLinear(input_dim, output_dim, args=args, layer=layer)
            elif args.enable_mask is True:
                self.linear = SparseLinearMulti_mask(input_dim, output_dim, args=args, layer=layer)
            else:
                self.linear = SparseLinear(input_dim, output_dim, args=args, layer=layer)

        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            if args.nmsparsity and args.enable_mask:
                self.linear = NMSparseMultiLinear(input_dim, hidden_dim, args=args, layer=layer)
            elif args.nmsparsity:
                self.linear = NMSparseLinear(input_dim, hidden_dim, args=args, layer=layer)
            elif args.enable_mask is True:
                self.linear = SparseLinearMulti_mask(input_dim, hidden_dim, args=args, layer=layer)
            else:
                self.linear = SparseLinear(input_dim, hidden_dim, args=args, layer=layer)

            for layer in range(num_layers - 2):
                if args.nmsparsity and args.enable_mask:
                    self.linear = NMSparseMultiLinear(
                        hidden_dim, hidden_dim, args=args, layer=layer
                    )
                elif args.nmsparsity:
                    self.linear = NMSparseLinear(hidden_dim, hidden_dim, args=args, layer=layer)
                elif args.enable_mask is True:
                    self.linear = SparseLinearMulti_mask(
                        hidden_dim, hidden_dim, args=args, layer=layer
                    )
                else:
                    self.linear = SparseLinear(hidden_dim, hidden_dim, args=args, layer=layer)

            if args.nmsparsity and args.enable_mask:
                self.linear = NMSparseMultiLinear(hidden_dim, output_dim, args=args, layer=layer)
            elif args.nmsparsity:
                self.linear = NMSparseLinear(hidden_dim, output_dim, args=args, layer=layer)
            elif args.enable_mask is True:
                self.linear = SparseLinearMulti_mask(hidden_dim, output_dim, args=args, layer=layer)
            else:
                self.linear = SparseLinear(hidden_dim, output_dim, args=args, layer=layer)

    def forward(self, x, threshold, sparsity=None):
        if self.linear_or_not:
            # If linear model
            return self.linear(x, threshold, sparsity)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.linears[i](h), threshold, sparsity)
            return self.linears[-1](h, threshold, sparsity)
