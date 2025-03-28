import copy
import os

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
# from littleballoffur import (
#     DegreeBasedSampler,
#     RandomEdgeSampler,
#     RandomNodeEdgeSampler,
#     RandomNodeSampler,
# )
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import csr_matrix
from torch import Tensor
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    Planetoid,
    Reddit,
    WebKB,
)
from torch_geometric.utils import (
    add_self_loops,
    from_networkx,
    from_scipy_sparse_matrix,
    remove_self_loops,
    to_networkx,
    to_undirected,
)
from torch_sparse import SparseTensor

from .ood import *


def load_ogbn(
    dataset="ogbn-arxiv", sampling=None, samplingtype=None, sparse_tensor=False
):
    dataset = PygNodePropPredDataset(
        # name=dataset, root="/ldisk/Shared/Datasets/GNNDataset"
        name=dataset, root="dataset"
    )
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    # print("x",data.x)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    if sparse_tensor:
        # data.edge_index = torch.sparse_coo_tensor(
        #     data.edge_index,
        #     torch.ones(data.edge_index.size(1)),
        #     size=(data.num_nodes, data.num_nodes),
        # )
        num_nodes = data.num_nodes
        row, col = data.edge_index
        edge_weight = torch.ones((col.size(0)))
        data.edge_index = SparseTensor(
            row=row,
            col=col,
            value=edge_weight,
            sparse_sizes=(num_nodes, num_nodes),
        )

    # if sampling is not None:
    #     graph = to_networkx(data, node_attrs=["x", "y"], to_undirected=True)
    #     if samplingtype == "RandomNodeSampler":
    #         number_of_nodes = int(sampling * graph.number_of_nodes())
    #         sampler = RandomNodeSampler(number_of_nodes=number_of_nodes)
    #         new_graph = sampler.sample(graph)
    #     elif samplingtype == "DegreeBasedSampler":
    #         number_of_nodes = int(sampling * graph.number_of_nodes())
    #         sampler = DegreeBasedSampler(number_of_nodes=number_of_nodes)
    #         new_graph = sampler.sample(graph)
    #     elif samplingtype == "RandomEdgeSampler":
    #         number_of_edges = int(sampling * graph.number_of_edges())
    #         sampler = RandomNodeEdgeSampler(number_of_edges=number_of_edges)
    #         new_graph = sampler.sample(graph)
    #         number_of_nodes = new_graph.number_of_nodes()
    #     else:
    #         print("wrong in sampling!")

    #     data1 = from_networkx(new_graph)
    #     if samplingtype == "RandomEdgeSampler":
    #         idxes = list(new_graph.nodes.keys())
    #         data1.x = data.x[idxes].contiguous()
    #         data1.y = data.y[idxes].contiguous()
    #     data = data1

    #     train_num = int(0.55 * number_of_nodes)
    #     test_num = int(0.3 * number_of_nodes)
    #     all_index = np.arange(number_of_nodes)
    #     train_index = np.random.choice(
    #         all_index, size=train_num, replace=False
    #     )
    #     index_remain = set(all_index) - set(train_index)
    #     index_remain_array = np.array(list(index_remain))
    #     test_index = np.random.choice(
    #         index_remain_array, size=test_num, replace=False
    #     )
    #     val_index = list(index_remain - set(test_index))
    #     split_idx = {
    #         "train": torch.tensor(train_index).long(),
    #         "test": torch.tensor(test_index).long(),
    #         "valid": torch.tensor(val_index).long(),
    #     }

    return data, split_idx


def random_coauthor_amazon_splits(data):
    # https://github.com/mengliu1998/DeeperGNN/blob/da1f21c40ec535d8b7a6c8127e461a1cd9eadac1/DeeperGNN/train_eval.py#L17
    num_classes, lcc = data.num_classes, data.lcc
    lcc_mask = None
    if lcc:  # select largest connected component
        data_nx = to_networkx(data)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(
            max(nx.connected_components(data_nx), key=len)
        )
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    def index_to_mask(index, size):
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask

    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


def manual_split_WebKB_Actor(data, which_split):
    # which_split take values from 0 to 9, type is int
    assert which_split in np.arange(10, dtype=int).tolist()

    data.train_mask = data.train_mask[:, which_split]
    data.val_mask = data.val_mask[:, which_split]
    data.test_mask = data.test_mask[:, which_split]
    return data


def change_split(data, dataset, which_split=0):
    if dataset in ["CoauthorCS", "CoauthorPhysics"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["AmazonComputers", "AmazonPhoto"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
        data = manual_split_WebKB_Actor(data, which_split)
    elif dataset == "ACTOR":
        data = manual_split_WebKB_Actor(data, which_split)
    else:
        data = data
    data.y = data.y.long()
    return data


def load_data(dataset, which_run, attack=None, attack_eps=0):
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data", dataset
    )

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        # path = os.path.join("/ldisk/Shared/Datasets/GNNDataset/", dataset)
        path = "dataset"
        data = Planetoid(
            path, dataset, split="public", transform=T.NormalizeFeatures()
        )[0]
        print("--------------------")
        # print(path)
    elif dataset in ["CoauthorCS", "CoauthorPhysics"]:
        data = Coauthor(path, dataset[8:], transform=T.NormalizeFeatures())[0]
        data.num_classes = int(max(data.y) + 1)
        data.lcc = False
        data = change_split(data, dataset, which_split=int(which_run // 10))
    elif dataset in ["AmazonComputers", "AmazonPhoto"]:
        data = Amazon(path, dataset[6:], transform=T.NormalizeFeatures())[0]
        data.num_classes = int(max(data.y) + 1)
        data.lcc = True
        data = change_split(data, dataset, which_split=int(which_run // 10))
    elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
        data = WebKB(path, dataset, transform=T.NormalizeFeatures())[0]
        data = change_split(data, dataset, which_split=int(which_run // 10))
    elif dataset == "ACTOR":
        data = Actor(path, transform=T.NormalizeFeatures())[0]
        data = change_split(data, dataset, which_split=int(which_run // 10))
    elif dataset == "Reddit":
        path = os.path.join("dataset", dataset)
        data = Reddit(path)[0]
        # data = change_split(data, dataset, which_split=int(which_run // 10))
    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")

    if attack is not None:
        if attack == "features":
            data, _ = perturb_features(data, ood_budget_per_graph=attack_eps)
        elif attack == "edges":
            data, _ = random_edge_perturbations(
                data, ood_budget_per_graph=attack_eps
            )

        else:
            raise Exception("not implemented attack type!")

    num_nodes = data.x.size(0)

    if dataset == "Reddit":
        num_nodes = data.num_nodes
        row, col = data.edge_index
        data.edge_index = SparseTensor(
            row=row, col=col, sparse_sizes=(num_nodes, num_nodes)
        )
    else:
        if isinstance(data.edge_index, Tensor):
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
            if isinstance(edge_index, tuple):
                data.edge_index = edge_index[0]
            else:
                data.edge_index = edge_index

        elif isinstance(data.edge_index, SparseTensor):
            data.edge_index = data.edge_index.set_diag()
            data.edge_index = data.edge_index.fill_value(1.0)
        else:
            raise ValueError(
                f"Unsupported edge_index type: {type(data.edge_index)}"
            )

    va_train_len = data.train_mask.sum()
    va_val_len = data.val_mask.sum()
    va_test_len = data.test_mask.sum()
    print("| # of train set : {}".format(va_train_len))
    print("| # of val set   : {}".format(va_val_len))
    print("| # of test set  : {}".format(va_test_len))

    return data.cuda()
