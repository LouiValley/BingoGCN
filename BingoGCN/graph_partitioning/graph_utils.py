from typing import Optional, Tuple, Union

import numpy as np
import pymetis
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter
from torch_scatter.utils import broadcast
from torch_sparse import SparseTensor


def generate_adj_list(adj_matrix):
    adj_list = []
    for i in range(adj_matrix.shape[0]):
        neighbors = (
            torch.nonzero(adj_matrix[i], as_tuple=False).squeeze().tolist()
        )
        neighbors = [neighbors] if isinstance(neighbors, int) else neighbors
        adj_list.append(neighbors)
    return adj_list


def generate_adj_list_ogbn(adj_matrix):
    if isinstance(adj_matrix, SparseTensor):
        row, col, _ = adj_matrix.coo()
        indices = torch.stack([row, col], dim=0).cpu().numpy()
    else:
        adj_matrix_coalesced = adj_matrix.coalesce()
        indices = adj_matrix_coalesced.indices().cpu().numpy()

    adj_list = [[] for _ in range(adj_matrix.size(0))]
    for i, j in indices.T:
        adj_list[i].append(j)

    return adj_list


# def generate_adj_list_ogbn(adj_matrix):
#     adj_matrix_coalesced = adj_matrix.coalesce()
#     indices = adj_matrix_coalesced.indices().cpu().numpy()
#     adj_list = [[] for _ in range(adj_matrix.shape[0])]
#     for i, j in indices.T:
#         adj_list[i].append(j)
#     return adj_list


def partition_graph(adj_list, n_parts):
    n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adj_list)
    return n_cuts, membership


def create_partitioned_adj_matrix(
    adj_matrix, sorted_indices, membership, part
):
    # 筛选出当前分区的节点索引
    part_nodes = sorted_indices[membership == part]

    # 创建分区的邻接矩阵
    part_adj_matrix = adj_matrix[part_nodes, :][:, part_nodes]
    return part_adj_matrix


def create_partitioned_adj_matrix_ogbn(
    adj_matrix, sorted_indices, membership, part
):
    part_nodes = sorted_indices[membership == part]
    part_nodes_set = set(part_nodes.tolist())
    node_to_new_idx = np.zeros(sorted_indices.max().item() + 1, dtype=np.int64)
    node_to_new_idx[part_nodes] = np.arange(len(part_nodes))

    indices = adj_matrix.coalesce().indices().cpu().numpy()
    values = adj_matrix.coalesce().values().cpu().numpy()

    # t1 = time.time()
    mask = np.isin(indices, list(part_nodes_set)).all(axis=0)
    new_indices = indices[:, mask]
    new_values = values[mask]

    new_indices[0] = torch.from_numpy(node_to_new_idx[new_indices[0]])
    new_indices[1] = torch.from_numpy(node_to_new_idx[new_indices[1]])
    # t2 = time.time()
    # elapsed_time = t2 - t1
    # print(f"vectorized operation in create_partitioned adj matrix ogbn{elapsed_time}")

    if not new_indices.size:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            [],
            [len(part_nodes), len(part_nodes)],
        )

    part_adj_matrix = torch.sparse_coo_tensor(
        indices=new_indices,
        values=torch.from_numpy(new_values),
        size=[len(part_nodes), len(part_nodes)],
    )
    return part_adj_matrix


# def add_remaining_self_loops(
#     edge_index: Union[Tensor, SparseTensor],
#     edge_attr: OptTensor = None,
#     fill_value: Union[float, Tensor, str] = None,
#     num_nodes: Optional[int] = None,
# ) -> Tuple[Tensor, OptTensor]:

#     if isinstance(edge_index, SparseTensor):
#         edge_index = edge_index.coo()[:2]

#     N = maybe_num_nodes(edge_index, num_nodes)
#     mask = edge_index[0] != edge_index[1]

#     loop_index = torch.arange(0, N, dtype=torch.long, device="cuda:0")
#     loop_index = loop_index.unsqueeze(0).repeat(2, 1)

#     if edge_attr is not None:
#         if fill_value is None:
#             loop_attr = edge_attr.new_full((N,) + edge_attr.size()[1:], 1.0)

#         elif isinstance(fill_value, (int, float)):
#             loop_attr = edge_attr.new_full(
#                 (N,) + edge_attr.size()[1:], fill_value
#             )
#         elif isinstance(fill_value, Tensor):
#             loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
#             if edge_attr.dim() != loop_attr.dim():
#                 loop_attr = loop_attr.unsqueeze(0)
#             sizes = [N] + [1] * (loop_attr.dim() - 1)
#             loop_attr = loop_attr.repeat(*sizes)

#         elif isinstance(fill_value, str):
#             loop_attr = scatter(
#                 edge_attr, edge_index[1], dim=0, dim_size=N, reduce=fill_value
#             )
#         else:
#             raise AttributeError("No valid 'fill_value' provided")

#         inv_mask = ~mask
#         loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]

#         edge_attr = torch.cat([edge_attr[mask], loop_attr], dim=0)

#     if isinstance(edge_index, tuple):
#         row, col = edge_index
#         edge_index = torch.stack(edge_index, dim=0)
#         edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
#     else:
#         edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

#     return edge_index, edge_attr


def add_remaining_self_loops(
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
            loop_attr = scatter(
                value, edge_index[1], dim=0, dim_size=N, reduce=fill_value
            )
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


def maybe_num_nodes(
    edge_index: Tensor, num_nodes: Optional[int] = None
) -> int:
    if num_nodes is not None:
        return num_nodes
    return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0


# def maybe_num_nodes(edge_index, num_nodes=None):
#     if num_nodes is not None:
#         return num_nodes
#     elif isinstance(edge_index, Tensor):
#         return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
#     else:
#         return max(edge_index.size(0), edge_index.size(1))


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def collect_inter_cluster_edges_modified(edge_index, membership, current_part):
    """
    收集与指定分区current_part连接的、属于不同分区的边。
    """
    inter_cluster_edges = []
    new_membership = membership.copy()

    # 遍历所有的边
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        # 检查边是否连接当前分区与不同分区
        if (
            membership[src] == current_part and membership[dst] != current_part
        ) or (
            membership[src] != current_part and membership[dst] == current_part
        ):
            # 如果是，添加这条边
            inter_cluster_edges.append((src.item(), dst.item()))

        if membership[src] == current_part and membership[dst] != current_part:
            new_membership[dst] = current_part
        if membership[src] != current_part and membership[dst] == current_part:
            new_membership[src] = current_part
    return inter_cluster_edges, new_membership


def add_inter_cluster_edges_to_adj_matrix(
    part_adj_matrix, inter_cluster_edges, part_indices, num_nodes
):
    """
    将跨分区的边添加到分区的邻接矩阵中。
    """
    # 创建一个新的邻接矩阵，考虑到外部节点
    enhanced_adj_matrix = torch.zeros(
        (num_nodes, num_nodes), dtype=part_adj_matrix.dtype
    )

    # 将原始分区的邻接矩阵复制到新的邻接矩阵中
    enhanced_adj_matrix[: len(part_indices), : len(part_indices)] = (
        part_adj_matrix
    )
    # 添加跨分区的边
    for src, dst in inter_cluster_edges:
        if len(np.where(part_indices == src)[0]) == 0:
            dst_idx = np.where(part_indices == dst)[0][0]
            enhanced_adj_matrix[src, dst_idx] = 1
            enhanced_adj_matrix[dst_idx, src] = 1

        else:
            src_idx = np.where(part_indices == src)[0][0]
            enhanced_adj_matrix[src_idx, dst] = 1
            enhanced_adj_matrix[dst, src_idx] = 1
    return enhanced_adj_matrix


def reorder_adj_matrix(adj_matrix, sorted_indices):
    return adj_matrix[sorted_indices, :][:, sorted_indices]


def count_zero_ratio(adj_matrix):
    total_elements = torch.numel(adj_matrix)
    non_zero_elements = torch.count_nonzero(adj_matrix)
    zero_elements = total_elements - non_zero_elements
    pre_sparse_intra_ratio = zero_elements.float() / total_elements
    return pre_sparse_intra_ratio


def adj_to_edge_index(adj_matrix):
    # 非零元素的位置表示存在边
    src, dst = adj_matrix.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def adj_to_edge_index_ogbn(adj_matrix):
    edge_index = adj_matrix.coalesce().indices()
    edge_index = SparseTensor(
        row=edge_index[0], col=edge_index[1], sparse_sizes=adj_matrix.shape
    )
    return edge_index
