# import numpy as np
import torch
from torch_sparse import SparseTensor

# def get_connected_nodes(
#     part_indices,
#     edge_index,
#     sparsity=0,
#     args=None,
# ):
#     if args.no_inter_cluster:
#         selected_nodes = part_indices.to(edge_index.device)
#         additional_nodes = torch.tensor(
#             [], dtype=torch.long, device=edge_index.device
#         )
#         num_additional_nodes = 0
#         num_nodes_to_delete = 0
#     else:
#         part_indices_tensor = part_indices.to(edge_index.device)
#         source_mask = torch.isin(edge_index[0], part_indices_tensor)
#         target_mask = torch.isin(edge_index[1], part_indices_tensor)
#         source_nodes = edge_index[0, ~source_mask & target_mask]
#         target_nodes = edge_index[1, source_mask & ~target_mask]
#         additional_nodes = torch.cat([source_nodes, target_nodes]).unique()
#         selected_nodes = torch.cat(
#             [part_indices_tensor, additional_nodes]
#         ).unique()
#         num_additional_nodes = len(additional_nodes)

#         # if args.random_sampling:
#         #     num_nodes_to_delete = int(len(additional_nodes) * sparsity)
#         #     delete_indices = torch.randperm(
#         #         len(additional_nodes), device=additional_nodes.device
#         #     )[:num_nodes_to_delete]
#         #     delete_nodes = additional_nodes[delete_indices]
#         # elif args.topk_sampling:
#         #     additional_nodes_count = torch.zeros(
#         #         edge_index.max() + 1,
#         #         dtype=torch.long,
#         #         device=edge_index.device,
#         #     )
#         #     additional_nodes_count.index_add_(
#         #         0,
#         #         edge_index[0],
#         #         torch.ones_like(edge_index[0], dtype=torch.long),
#         #     )
#         #     additional_nodes_count.index_add_(
#         #         0,
#         #         edge_index[1],
#         #         torch.ones_like(edge_index[1], dtype=torch.long),
#         #     )
#         #     additional_nodes_count = additional_nodes_count[additional_nodes]
#         #     num_nodes_to_delete = int(len(additional_nodes) * sparsity)
#         #     _, delete_indices = additional_nodes_count.topk(
#         #         num_nodes_to_delete, largest=False
#         #     )
#         #     delete_nodes = additional_nodes[delete_indices]
#         # elif args.topk_pruning:
#         #     additional_nodes_count = torch.zeros(
#         #         edge_index.max() + 1,
#         #         dtype=torch.long,
#         #         device=edge_index.device,
#         #     )
#         #     additional_nodes_count.index_add_(
#         #         0,
#         #         edge_index[0],
#         #         torch.ones_like(edge_index[0], dtype=torch.long),
#         #     )
#         #     additional_nodes_count.index_add_(
#         #         0,
#         #         edge_index[1],
#         #         torch.ones_like(edge_index[1], dtype=torch.long),
#         #     )
#         #     additional_nodes_count = additional_nodes_count[additional_nodes]
#         #     num_nodes_to_delete = int(len(additional_nodes) * sparsity)
#         #     _, delete_indices = additional_nodes_count.topk(
#         #         num_nodes_to_delete, largest=True
#         #     )
#         #     delete_nodes = additional_nodes[delete_indices]
#         # else:
#          delete_nodes = torch.tensor(
#              [], dtype=torch.long, device=additional_nodes.device
#          )
#          num_nodes_to_delete = 0

#         selected_nodes = selected_nodes[
#             ~torch.isin(selected_nodes, delete_nodes)
#         ]

#     original_num_nodes = len(part_indices)
#     final_num_nodes = len(selected_nodes)

#     return (
#         selected_nodes,
#         additional_nodes,
#         original_num_nodes,
#         num_additional_nodes - num_nodes_to_delete,
#         final_num_nodes,
#     )


# def get_connected_nodes(
#     part_indices,
#     edge_index,
#     sparsity=0,
#     args=None,
# ):
#     device = "cuda:0"
#     if args.no_inter_cluster:
#         selected_nodes = part_indices.to(device)
#         additional_nodes = torch.tensor([], dtype=torch.long, device=device)
#         num_additional_nodes = 0
#         num_nodes_to_delete = 0
#         num_outgoing_nodes = 0
#     else:
#         part_indices_tensor = part_indices.to(device)
#         source_mask = torch.isin(edge_index[0], part_indices_tensor)
#         target_mask = torch.isin(edge_index[1], part_indices_tensor)
#         source_nodes = edge_index[0, ~source_mask & target_mask]
#         target_nodes = edge_index[1, source_mask & ~target_mask]
#         additional_nodes = torch.cat([source_nodes, target_nodes]).unique()
#         selected_nodes = torch.cat(
#             [part_indices_tensor, additional_nodes]
#         ).unique()
#         num_additional_nodes = len(additional_nodes)
#         delete_nodes = torch.tensor([], dtype=torch.long, device=device)
#         num_nodes_to_delete = 0

#         selected_nodes = selected_nodes[
#             ~torch.isin(selected_nodes, delete_nodes)
#         ]

#         # Correctly identify outgoing nodes
#         outgoing_nodes_mask = (
#             torch.isin(edge_index[0], part_indices_tensor)
#             & ~torch.isin(edge_index[1], part_indices_tensor)
#         ) | (
#             torch.isin(edge_index[1], part_indices_tensor)
#             & ~torch.isin(edge_index[0], part_indices_tensor)
#         )
#         outgoing_nodes = torch.cat(
#             [
#                 edge_index[0, outgoing_nodes_mask],
#                 edge_index[1, outgoing_nodes_mask],
#             ]
#         ).unique()
#         outgoing_nodes = outgoing_nodes[
#             torch.isin(outgoing_nodes, part_indices_tensor)
#         ]
#         num_outgoing_nodes = len(outgoing_nodes)

#     original_num_nodes = len(part_indices)
#     final_num_nodes = len(selected_nodes)

#     new_part_indices = selected_nodes
#     additional_nodes_indices = additional_nodes
#     num_intra_nodes = original_num_nodes
#     num_inter_nodes = num_additional_nodes - num_nodes_to_delete
#     num_intra_and_inter_nodes = final_num_nodes

#     return (
#         new_part_indices,
#         additional_nodes_indices,
#         num_intra_nodes,
#         num_inter_nodes,
#         num_intra_and_inter_nodes,
#         num_outgoing_nodes,
#     )


# def get_connected_nodes(
#     part_indices,
#     edge_index,
#     sparsity=0,
#     args=None,
# ):
#     device = "cpu"
#     part_indices_tensor = part_indices.to(device)

#     if isinstance(edge_index, SparseTensor):
#         row, col, value = edge_index.coo()
#         edge_index_tensor = torch.stack([row, col], dim=0)
#     else:
#         edge_index_tensor = edge_index

#     if args and args.no_inter_cluster:
#         selected_nodes = part_indices_tensor
#         additional_nodes = torch.tensor([], dtype=torch.long, device=device)
#         num_additional_nodes = 0
#         num_nodes_to_delete = 0
#         num_outgoing_nodes = 0
#     else:
#         source_mask = torch.isin(edge_index_tensor[0], part_indices_tensor)
#         target_mask = torch.isin(edge_index_tensor[1], part_indices_tensor)
#         source_nodes = edge_index_tensor[0, ~source_mask & target_mask]
#         target_nodes = edge_index_tensor[1, source_mask & ~target_mask]
#         additional_nodes = torch.cat([source_nodes, target_nodes]).unique()
#         selected_nodes = torch.cat(
#             [part_indices_tensor, additional_nodes]
#         ).unique()
#         num_additional_nodes = len(additional_nodes)
#         delete_nodes = torch.tensor([], dtype=torch.long, device=device)
#         num_nodes_to_delete = 0

#         selected_nodes = selected_nodes[
#             ~torch.isin(selected_nodes, delete_nodes)
#         ]

#         # Correctly identify outgoing nodes
#         outgoing_nodes_mask = (
#             torch.isin(edge_index_tensor[0], part_indices_tensor)
#             & ~torch.isin(edge_index_tensor[1], part_indices_tensor)
#         ) | (
#             torch.isin(edge_index_tensor[1], part_indices_tensor)
#             & ~torch.isin(edge_index_tensor[0], part_indices_tensor)
#         )
#         outgoing_nodes = torch.cat(
#             [
#                 edge_index_tensor[0, outgoing_nodes_mask],
#                 edge_index_tensor[1, outgoing_nodes_mask],
#             ]
#         ).unique()
#         outgoing_nodes = outgoing_nodes[
#             torch.isin(outgoing_nodes, part_indices_tensor)
#         ]
#         num_outgoing_nodes = len(outgoing_nodes)

#     original_num_nodes = len(part_indices)
#     final_num_nodes = len(selected_nodes)

#     new_part_indices = selected_nodes
#     additional_nodes_indices = additional_nodes
#     num_intra_nodes = original_num_nodes
#     num_inter_nodes = num_additional_nodes - num_nodes_to_delete
#     num_intra_and_inter_nodes = final_num_nodes

#     return (
#         new_part_indices,
#         additional_nodes_indices,
#         num_intra_nodes,
#         num_inter_nodes,
#         num_intra_and_inter_nodes,
#         num_outgoing_nodes,
#     )


def get_connected_nodes_random_topk(
    part_indices,
    edge_index,
    sparsity=0,
    args=None,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    part_indices_tensor = part_indices.to(device)

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        edge_index_tensor = torch.stack([row, col], dim=0).to(device)
        del row, col
    else:
        edge_index_tensor = edge_index.to(device)

    if args.no_inter_cluster:
        selected_nodes = part_indices_tensor
        additional_nodes = torch.tensor([], dtype=torch.long, device=device)
        num_additional_nodes = 0
        num_nodes_to_delete = 0
        num_outgoing_nodes = 0
    else:
        source_mask = torch.isin(edge_index_tensor[0], part_indices_tensor)
        target_mask = torch.isin(edge_index_tensor[1], part_indices_tensor)
        source_nodes = edge_index_tensor[0][~source_mask & target_mask]
        target_nodes = edge_index_tensor[1][source_mask & ~target_mask]
        additional_nodes = torch.cat([source_nodes, target_nodes]).unique()
        selected_nodes = torch.cat(
            [part_indices_tensor, additional_nodes]
        ).unique()
        num_additional_nodes = len(additional_nodes)

        if args.random_sampling:
            num_nodes_to_delete = int(len(additional_nodes) * sparsity)
            delete_indices = torch.randperm(
                len(additional_nodes), device=additional_nodes.device
            )[:num_nodes_to_delete]
            delete_nodes = additional_nodes[delete_indices]
        elif args.topk_sampling:
            additional_nodes_count = torch.zeros(
                edge_index_tensor.max() + 1,
                dtype=torch.long,
                device=edge_index_tensor.device,
            )
            additional_nodes_count.index_add_(
                0,
                edge_index_tensor[0],
                torch.ones_like(edge_index_tensor[0], dtype=torch.long),
            )
            additional_nodes_count.index_add_(
                0,
                edge_index_tensor[1],
                torch.ones_like(edge_index_tensor[1], dtype=torch.long),
            )
            additional_nodes_count = additional_nodes_count[additional_nodes]
            num_nodes_to_delete = int(len(additional_nodes) * sparsity)
            _, delete_indices = additional_nodes_count.topk(
                num_nodes_to_delete, largest=False
            )
            delete_nodes = additional_nodes[delete_indices]
        else:
            delete_nodes = torch.tensor(
                [], dtype=torch.long, device=additional_nodes.device
            )
            num_nodes_to_delete = 0

        selected_nodes = selected_nodes[
            ~torch.isin(selected_nodes, delete_nodes)
        ]

        outgoing_nodes_mask = (
            torch.isin(edge_index_tensor[0], part_indices_tensor)
            & ~torch.isin(edge_index_tensor[1], part_indices_tensor)
        ) | (
            torch.isin(edge_index_tensor[1], part_indices_tensor)
            & ~torch.isin(edge_index_tensor[0], part_indices_tensor)
        )
        outgoing_nodes = torch.cat(
            [
                edge_index_tensor[0][outgoing_nodes_mask],
                edge_index_tensor[1][outgoing_nodes_mask],
            ]
        ).unique()
        outgoing_nodes = outgoing_nodes[
            torch.isin(outgoing_nodes, part_indices_tensor)
        ]
        num_outgoing_nodes = len(outgoing_nodes)

    original_num_nodes = len(part_indices)
    final_num_nodes = len(selected_nodes)

    new_part_indices = selected_nodes
    additional_nodes_indices = additional_nodes
    num_intra_nodes = original_num_nodes
    num_inter_nodes = num_additional_nodes - num_nodes_to_delete
    num_intra_and_inter_nodes = final_num_nodes

    if not args.no_inter_cluster:
        del part_indices_tensor, edge_index_tensor, source_mask, target_mask
        del (
            source_nodes,
            target_nodes,
            delete_nodes,
            outgoing_nodes_mask,
            outgoing_nodes,
        )
    if device == "cuda:0":
        torch.cuda.empty_cache()

    return (
        new_part_indices,
        additional_nodes_indices,
        num_intra_nodes,
        num_inter_nodes,
        num_intra_and_inter_nodes,
        num_outgoing_nodes,
    )


def get_connected_nodes(
    part_indices,
    edge_index,
    sparsity=0,
    args=None,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    part_indices_tensor = part_indices.to(device)

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        edge_index_tensor = torch.stack([row, col], dim=0).to(device)
        del row, col
    else:
        edge_index_tensor = edge_index.to(device)

    # if args and args.no_inter_cluster:
    if args.no_inter_cluster:
        selected_nodes = part_indices_tensor
        additional_nodes = torch.tensor([], dtype=torch.long, device=device)
        num_additional_nodes = 0
        num_nodes_to_delete = 0
        num_outgoing_nodes = 0
    else:
        source_mask = torch.isin(edge_index_tensor[0], part_indices_tensor)
        target_mask = torch.isin(edge_index_tensor[1], part_indices_tensor)
        source_nodes = edge_index_tensor[0][~source_mask & target_mask]
        target_nodes = edge_index_tensor[1][source_mask & ~target_mask]
        additional_nodes = torch.cat([source_nodes, target_nodes]).unique()
        selected_nodes = torch.cat(
            [part_indices_tensor, additional_nodes]
        ).unique()
        num_additional_nodes = len(additional_nodes)
        delete_nodes = torch.tensor([], dtype=torch.long, device=device)
        num_nodes_to_delete = 0

        selected_nodes = selected_nodes[
            ~torch.isin(selected_nodes, delete_nodes)
        ]

        outgoing_nodes_mask = (
            torch.isin(edge_index_tensor[0], part_indices_tensor)
            & ~torch.isin(edge_index_tensor[1], part_indices_tensor)
        ) | (
            torch.isin(edge_index_tensor[1], part_indices_tensor)
            & ~torch.isin(edge_index_tensor[0], part_indices_tensor)
        )
        outgoing_nodes = torch.cat(
            [
                edge_index_tensor[0][outgoing_nodes_mask],
                edge_index_tensor[1][outgoing_nodes_mask],
            ]
        ).unique()
        outgoing_nodes = outgoing_nodes[
            torch.isin(outgoing_nodes, part_indices_tensor)
        ]
        num_outgoing_nodes = len(outgoing_nodes)

    original_num_nodes = len(part_indices)
    final_num_nodes = len(selected_nodes)

    new_part_indices = selected_nodes
    additional_nodes_indices = additional_nodes
    num_intra_nodes = original_num_nodes
    num_inter_nodes = num_additional_nodes - num_nodes_to_delete
    num_intra_and_inter_nodes = final_num_nodes

    if not args.no_inter_cluster:
        del part_indices_tensor, edge_index_tensor, source_mask, target_mask
        del (
            source_nodes,
            target_nodes,
            delete_nodes,
            outgoing_nodes_mask,
            outgoing_nodes,
        )
    if device == "cuda:0":
        torch.cuda.empty_cache()

    return (
        new_part_indices,
        additional_nodes_indices,
        num_intra_nodes,
        num_inter_nodes,
        num_intra_and_inter_nodes,
        num_outgoing_nodes,
    )


# def get_connected_nodes(
#     part_indices,
#     edge_index,
#     sparsity=0,
#     args=None,
# ):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     part_indices_tensor = part_indices.to(device)

#     if isinstance(edge_index, SparseTensor):
#         row, col, value = edge_index.coo()
#         edge_index_tensor = torch.stack([row, col], dim=0).to(device)
#     else:
#         edge_index_tensor = edge_index.to(device)

#     if args and args.no_inter_cluster:
#         selected_nodes = part_indices_tensor
#         additional_nodes = torch.tensor([], dtype=torch.long, device=device)
#         num_additional_nodes = 0
#         num_nodes_to_delete = 0
#         num_outgoing_nodes = 0
#     else:
#         source_mask = torch.isin(edge_index_tensor[0], part_indices_tensor)
#         target_mask = torch.isin(edge_index_tensor[1], part_indices_tensor)
#         source_nodes = edge_index_tensor[0, ~source_mask & target_mask]
#         target_nodes = edge_index_tensor[1, source_mask & ~target_mask]
#         additional_nodes = torch.cat([source_nodes, target_nodes]).unique()
#         selected_nodes = torch.cat(
#             [part_indices_tensor, additional_nodes]
#         ).unique()
#         num_additional_nodes = len(additional_nodes)
#         delete_nodes = torch.tensor([], dtype=torch.long, device=device)
#         num_nodes_to_delete = 0

#         selected_nodes = selected_nodes[
#             ~torch.isin(selected_nodes, delete_nodes)
#         ]

#         outgoing_nodes_mask = (
#             torch.isin(edge_index_tensor[0], part_indices_tensor)
#             & ~torch.isin(edge_index_tensor[1], part_indices_tensor)
#         ) | (
#             torch.isin(edge_index_tensor[1], part_indices_tensor)
#             & ~torch.isin(edge_index_tensor[0], part_indices_tensor)
#         )
#         outgoing_nodes = torch.cat(
#             [
#                 edge_index_tensor[0, outgoing_nodes_mask],
#                 edge_index_tensor[1, outgoing_nodes_mask],
#             ]
#         ).unique()
#         outgoing_nodes = outgoing_nodes[
#             torch.isin(outgoing_nodes, part_indices_tensor)
#         ]
#         num_outgoing_nodes = len(outgoing_nodes)

#     original_num_nodes = len(part_indices)
#     final_num_nodes = len(selected_nodes)

#     new_part_indices = selected_nodes.to("cpu")
#     additional_nodes_indices = additional_nodes.to("cpu")
#     num_intra_nodes = original_num_nodes
#     num_inter_nodes = num_additional_nodes - num_nodes_to_delete
#     num_intra_and_inter_nodes = final_num_nodes

#     return (
#         new_part_indices,
#         additional_nodes_indices,
#         num_intra_nodes,
#         num_inter_nodes,
#         num_intra_and_inter_nodes,
#         num_outgoing_nodes,
#     )


def filter_edges(edge_index, new_part_indices, edge_weight=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    new_part_indices_tensor = new_part_indices.to(device)

    if isinstance(edge_index, SparseTensor):
        row, col, value = edge_index.coo()
        edge_index_tensor = torch.stack([row, col], dim=0).to(device)
        edge_weight = value.to(device)
    else:
        edge_index_tensor = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

    source_mask = torch.isin(edge_index_tensor[0], new_part_indices_tensor)
    target_mask = torch.isin(edge_index_tensor[1], new_part_indices_tensor)
    mask = source_mask & target_mask

    filtered_edge_index = edge_index_tensor[:, mask]
    filtered_edge_weight = (
        edge_weight[mask] if edge_weight is not None else None
    )

    if isinstance(edge_index, SparseTensor):
        filtered_edge_index = SparseTensor(
            row=filtered_edge_index[0].to(device),
            col=filtered_edge_index[1].to(device),
            value=(
                filtered_edge_weight.to(device)
                if filtered_edge_weight is not None
                else None
            ),
            sparse_sizes=edge_index.sparse_sizes(),
        )
        filtered_edge_weight = (
            filtered_edge_weight.to(device)
            if filtered_edge_weight is not None
            else None
        )
    else:
        filtered_edge_index = filtered_edge_index.to(device)
        filtered_edge_weight = (
            filtered_edge_weight.to(device)
            if filtered_edge_weight is not None
            else None
        )

    return filtered_edge_index, filtered_edge_weight


def reindex_nodes(edge_index, connected_nodes_indices):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    connected_nodes_indices = connected_nodes_indices.to(device)

    unique_nodes, new_indices = torch.unique(
        connected_nodes_indices, return_inverse=True
    )

    if isinstance(edge_index, SparseTensor):
        row, col, value = edge_index.coo()
        edge_index_tensor = torch.stack([row, col], dim=0).to(device)
        value = value.to(device)
    else:
        edge_index_tensor = edge_index.to(device)

    old_to_new_idx = torch.empty(
        connected_nodes_indices.max() + 1,
        dtype=torch.long,
        device=edge_index_tensor.device,
    )
    old_to_new_idx[unique_nodes] = torch.arange(
        len(unique_nodes), device=edge_index_tensor.device
    )
    reindexed_edge_index = old_to_new_idx[edge_index_tensor]

    if isinstance(edge_index, SparseTensor):
        reindexed_edge_index = SparseTensor(
            row=reindexed_edge_index[0].to(device),
            col=reindexed_edge_index[1].to(device),
            value=value.to(device),
            sparse_sizes=(
                connected_nodes_indices.size(0),
                connected_nodes_indices.size(0),
            ),
            # sparse_sizes=edge_index.sparse_sizes(),
        )
        unique_nodes = unique_nodes.to(device)
    else:
        reindexed_edge_index = reindexed_edge_index.to(device)
        unique_nodes = unique_nodes.to(device)

    return reindexed_edge_index, unique_nodes


# class OnlineKMeans:
#     def __init__(
#         self,
#         n_clusters,
#         device,
#         init="random",
#         random_state=0,
#         initial_centroids=None,
#         distance_type="l2",
#     ):
#         self.n_clusters = n_clusters
#         self.device = device
#         self.init = init
#         self.random_state = random_state
#         self.cluster_centers = initial_centroids
#         if initial_centroids is not None:
#             self.cluster_centers = initial_centroids.to(device)
#         self.cluster_counts = None
#         self.distance_type = distance_type

# def compute_distance(self, features):
#     features = features.to(
#         self.device
#     )  # featuresを指定されたデバイスに移動
#     if self.distance_type == "l2":
#         return torch.sum(
#             (features[:, None, :] - self.cluster_centers[None, :, :]) ** 2,
#             dim=2,
#         )
#     elif self.distance_type == "l1":
#         return torch.sum(
#             torch.abs(
#                 features[:, None, :] - self.cluster_centers[None, :, :]
#             ),
#             dim=2,
#         )
#     elif self.distance_type == "dot":
#         return -torch.matmul(features, self.cluster_centers.T)
#     elif self.distance_type == "cosine":
#         features_norm = features / torch.norm(
#             features, dim=1, keepdim=True
#         )
#         centers_norm = self.cluster_centers / torch.norm(
#             self.cluster_centers, dim=1, keepdim=True
#         )
#         return 1 - torch.matmul(features_norm, centers_norm.T)
#     else:
#         raise ValueError("Unsupported distance type")

# def fit(self, features):
#     features = features.to(
#         self.device
#     )  # featuresを指定されたデバイスに移動
#     if self.cluster_centers is None:
#         self.initialize_centers(features)

#     if self.cluster_counts is None:
#         self.cluster_counts = torch.zeros(
#             self.n_clusters, dtype=torch.long, device=self.device
#         )

#     distances = self.compute_distance(features)
#     cluster_labels = torch.argmin(distances, dim=1)

#     # クラスタラベルごとに特徴を集約して平均を計算
#     for k in range(self.n_clusters):
#         mask = cluster_labels == k
#         self.cluster_counts[k] = mask.sum()
#         if self.cluster_counts[k] > 0:
#             self.cluster_centers[k] = torch.mean(features[mask], dim=0)

#     # ブロードキャスティングを使用して全てのクラスタを同時に更新
#     for k in range(self.n_clusters):
#         mask = cluster_labels == k
#         self.cluster_counts[k] = mask.sum()
#         if self.cluster_counts[k] > 0:
#             self.cluster_centers[k] = (
#                 torch.sum(features * mask[:, None], dim=0)
#                 / self.cluster_counts[k]
#             )

#     return cluster_labels


class HierarchicalOnlineKMeans:
    def __init__(
        self,
        n_clusters,
        n_cofc,
        device,
        init="random",
        random_state=0,
        distance_type="cosine",
        layer=None,
    ):
        self.n_clusters = n_clusters
        self.n_cofc = n_cofc
        self.device = device
        self.init = init
        self.random_state = random_state
        self.distance_type = distance_type
        self.cluster_centers = None
        self.cofc_centers = None
        self.cluster_counts = None
        self.cofc_counts = None
        self.cofc_to_centroids = {}  # CofC to centroid indices
        self.layer = layer

    def initialize_centers_randomly(self, features):
        # torch.manual_seed(self.random_state)
        indices = torch.randperm(features.size(0))[: self.n_clusters]
        self.cluster_centers = features[indices].clone()

    def initialize_centers_from_first(self, features):
        self.cluster_centers = features[: self.n_clusters].clone()

    def compute_distance(self, centers, point):
        # 1次元の場合は次元を追加する
        if centers.dim() == 1:
            centers = centers.unsqueeze(0)
        if point.dim() == 1:
            point = point.unsqueeze(0)

        if self.distance_type == "l2":
            return torch.sum((centers - point) ** 2, dim=1)
        elif self.distance_type == "l1":
            return torch.sum(torch.abs(centers - point), dim=1)
        elif self.distance_type == "dot":
            return -torch.matmul(centers, point)
        elif self.distance_type == "cosine":
            point_norm = point / torch.norm(point)
            if centers.dim() == 1:
                centers = centers.unsqueeze(0)
            centers_norm = centers / torch.norm(centers, dim=1, keepdim=True)
            return 1 - torch.matmul(
                centers_norm, point_norm.unsqueeze(-1)
            ).squeeze(-1)
        else:
            raise ValueError("Unsupported distance type")

    def online_kmeans_initialization(self, features, n_centers):
        # torch.manual_seed(self.random_state)
        self.cofc_features = features[:n_centers].clone()
        # indices = torch.randperm(features.size(0))[:n_centers]
        # self.cofc_features = features[indices].clone()
        self.cofc_cluster_labels = torch.zeros(
            features.size(0), dtype=torch.long, device=self.device
        )
        self.cofc_cluster_counts = torch.zeros(
            self.n_cofc, dtype=torch.long, device=self.device
        )

        for i in range(features.size(0)):
            if i < n_centers:
                cluster_id = i
                self.cofc_cluster_labels[i] = cluster_id
                self.cofc_cluster_counts[cluster_id] += 1
            else:
                point = self.cluster_centers[i]
                distances = self.compute_distance(self.cofc_features, point)
                cluster_id = torch.argmin(distances)
                self.cofc_cluster_labels[i] = cluster_id
                self.cofc_cluster_counts[cluster_id] += 1

                lr = 1.0 / self.cofc_cluster_counts[cluster_id]
                self.cofc_features[cluster_id] += lr * (
                    point - self.cofc_features[cluster_id]
                )

    def initialize_clusters(self, features):
        self.initialize_centers_from_first(features)
        self.online_kmeans_initialization(self.cluster_centers, self.n_cofc)

    def fit(self, features):
        if self.cluster_centers is None or self.cofc_centers is None:
            self.initialize_clusters(features)

        self.cluster_labels = torch.zeros(
            features.size(0), dtype=torch.long, device=self.device
        )
        self.cluster_counts = torch.zeros(
            self.n_clusters, dtype=torch.long, device=self.device
        )

        for i in range(features.size(0)):
            if i < self.n_clusters:
                cluster_id = i
                self.cluster_labels[i] = cluster_id
                self.cluster_counts[cluster_id] += 1
            else:
                point = features[i]
                cofc_distances = self.compute_distance(
                    self.cofc_features, point
                )
                cofc_id = torch.argmin(cofc_distances)

                relevant_centroids_indices = (
                    (self.cofc_cluster_labels == cofc_id).nonzero().squeeze()
                )
                relevant_centroids = self.cluster_centers[
                    relevant_centroids_indices
                ]

                cluster_distances = self.compute_distance(
                    relevant_centroids, point
                )
                relative_cluster_id = torch.argmin(cluster_distances)
                if relevant_centroids_indices.dim() == 0:
                    relevant_centroids_indices = [
                        relevant_centroids_indices.item()
                    ]
                cluster_id = relevant_centroids_indices[relative_cluster_id]
                self.cluster_labels[i] = cluster_id

                self.cluster_counts[cluster_id] += 1
                lr_cluster = 1.0 / self.cluster_counts[cluster_id]
                update_values = lr_cluster * (
                    point - self.cluster_centers[cluster_id]
                )
                self.cluster_centers[cluster_id] += update_values

                lr_cofc = 1.0 / self.cofc_cluster_counts[cofc_id]
                self.cofc_features[cofc_id] += lr_cofc * update_values

        return (
            self.cluster_labels,
            self.cofc_features,
            self.cofc_cluster_labels,
        )


class OnlineKMeans:
    def __init__(
        self,
        n_clusters,
        device,
        init="random",
        random_state=0,
        initial_centroids=None,
        distance_type="l2",
    ):
        self.n_clusters = n_clusters
        self.device = device
        self.init = init
        self.random_state = random_state
        self.cluster_centers = initial_centroids
        self.cluster_counts = None
        self.distance_type = distance_type

    def compute_distance(self, point):
        if self.distance_type == "l2":
            return torch.sum((self.cluster_centers - point) ** 2, dim=1)
        elif self.distance_type == "l1":
            return torch.sum(torch.abs(self.cluster_centers - point), dim=1)
        elif self.distance_type == "dot":
            return -torch.matmul(self.cluster_centers, point)
        elif self.distance_type == "cosine":
            point_norm = point / torch.norm(point)
            centers_norm = self.cluster_centers / torch.norm(
                self.cluster_centers, dim=1, keepdim=True
            )
            return 1 - torch.matmul(centers_norm, point_norm)
        elif self.distance_type == "cosine_no_sqrt":
            point_abs = point / torch.sum(point**2)
            centers_abs = self.cluster_centers / torch.sum(
                self.cluster_centers**2, dim=1, keepdim=True
            )
            return 1 - torch.matmul(centers_abs, point_abs)
        else:
            raise ValueError("Unsupported distance type")

    def fit(self, features):
        if self.cluster_centers is None:
            self.initialize_centers(features)

        if self.cluster_counts is None:
            self.cluster_counts = torch.zeros(
                self.n_clusters, dtype=torch.long, device=self.device
            )

        cluster_labels = torch.zeros(
            features.size(0), dtype=torch.long, device=self.device
        )

        for i in range(features.size(0)):
            point = features[i]
            distances = self.compute_distance(point)
            cluster_id = torch.argmin(distances)
            cluster_labels[i] = cluster_id

            self.cluster_counts[cluster_id] += 1

            # moving average
            lr = 1.0 / self.cluster_counts[cluster_id]
            self.cluster_centers[cluster_id] += lr * (
                point - self.cluster_centers[cluster_id]
            )

        return cluster_labels, None

    def initialize_centers(self, features):
        if self.cluster_centers is not None:
            return

        if self.init == "random":
            self.initialize_centers_randomly(features)
        elif self.init == "kmeans++":
            self.initialize_centers_kmeans_plusplus(features)
        else:
            raise ValueError(f"Invalid initialization method: {self.init}")

    def initialize_centers_randomly(self, features):
        # torch.manual_seed(self.random_state)
        indices = torch.randperm(features.size(0))[: self.n_clusters]
        self.cluster_centers = features[indices].clone()
        self.cluster_counts = torch.zeros(
            self.n_clusters, dtype=torch.long, device=self.device
        )

    # def compute_distances(self, features):
    #     if self.distance_type == "l2":
    #         diff = self.cluster_centers.unsqueeze(1) - features.unsqueeze(0)
    #         return torch.sum(diff**2, dim=2)
    #     elif self.distance_type == "l1":
    #         diff = torch.abs(self.cluster_centers.unsqueeze(1) - features.unsqueeze(0))
    #         return torch.sum(diff, dim=2)
    #     elif self.distance_type == "dot":
    #         return -torch.matmul(self.cluster_centers, features.t())
    #     elif self.distance_type == "cosine":
    #         features_norm = features / torch.norm(features, dim=1, keepdim=True)
    #         centers_norm = self.cluster_centers / torch.norm(self.cluster_centers, dim=1, keepdim=True)
    #         return 1 - torch.matmul(centers_norm, features_norm.t()).t()
    #     else:
    #         raise ValueError("Unsupported distance type")

    # def fit(self, features, batch_size=512):
    #     if self.cluster_centers is None:
    #         self.initialize_centers(features)

    #     if self.cluster_counts is None:
    #         self.cluster_counts = torch.zeros(self.n_clusters, dtype=torch.long, device=self.device)

    #     num_samples = features.shape[0]
    #     cluster_labels = torch.zeros(num_samples, dtype=torch.long, device=self.device)

    #     for start_idx in range(0, num_samples, batch_size):
    #         end_idx = min(start_idx + batch_size, num_samples)
    #         batch_features = features[start_idx:end_idx]
    #         distances = self.compute_distances(batch_features)
    #         cluster_ids = torch.argmin(distances, dim=0)

    #         for i, idx in enumerate(range(start_idx, end_idx)):
    #             cluster_id = cluster_ids[i]
    #             cluster_labels[idx] = cluster_id
    #             self.cluster_counts[cluster_id] += 1
    #             lr = 1.0 / self.cluster_counts[cluster_id]
    #             self.cluster_centers[cluster_id] += lr * (batch_features[i] - self.cluster_centers[cluster_id])

    #         torch.cuda.empty_cache()

    #     return cluster_labels, None

    def transform(self, features, approximation="center"):
        cluster_labels = self.fit(features)

        if approximation == "center":
            clustered_features = self.cluster_centers[cluster_labels]
        elif approximation == "average":
            clustered_features = torch.zeros_like(features)
            for cluster_id in range(self.n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_features = features[cluster_mask]
                cluster_mean = cluster_features.mean(dim=0)
                clustered_features[cluster_mask] = cluster_mean

        return clustered_features

    # def initialize_centers_kmeans_plusplus(self, features):
    #     n_samples, n_features = features.size()
    #     centers = torch.empty(
    #         (self.n_clusters, n_features),
    #         dtype=features.dtype,
    #         device=self.device,
    #     )
    #     n_local_trials = 2 + int(np.log(self.n_clusters))

    #     # Pick first center randomly
    #     torch.manual_seed(self.random_state)
    #     center_id = torch.randint(n_samples, size=(1,)).item()
    #     centers[0] = features[center_id].clone()

    #     # Initialize list of closest distances and calculate current potential
    #     closest_dist_sq = torch.sum((features - centers[0]) ** 2, dim=1)
    #     current_pot = closest_dist_sq.sum()

    #     # Pick the remaining n_clusters-1 points
    #     for c in range(1, self.n_clusters):
    #         # Choose center candidates by sampling with probability proportional
    #         # to the squared distance to the closest existing center
    #         rand_vals = (
    #             torch.rand(n_local_trials, device=self.device) * current_pot
    #         )
    #         candidate_ids = torch.searchsorted(
    #             torch.cumsum(closest_dist_sq, dim=0), rand_vals
    #         )
    #         candidate_ids = torch.clamp(
    #             candidate_ids, max=closest_dist_sq.size(0) - 1
    #         )

    #         # Compute distances to center candidates
    #         distance_to_candidates = torch.sum(
    #             (features[candidate_ids] - features.unsqueeze(1)) ** 2, dim=2
    #         )

    #         # Update closest distances squared and potential for each candidate
    #         closest_dist_sq = torch.min(
    #             closest_dist_sq, distance_to_candidates
    #         )
    #         candidates_pot = distance_to_candidates.sum(dim=1)

    #         # Decide which candidate is the best
    #         best_candidate = torch.argmin(candidates_pot).item()
    #         current_pot = candidates_pot[best_candidate]
    #         closest_dist_sq = distance_to_candidates[best_candidate]

    #         # Permanently add best center candidate found in local tries
    #         centers[c] = features[candidate_ids[best_candidate]].clone()

    #     self.cluster_centers = centers
    #     self.cluster_counts = torch.zeros(
    #         self.n_clusters, dtype=torch.long, device=self.device
    #     )


class StreamingKMeans:
    def __init__(self, n_clusters, n_parts, device):
        self.n_clusters = n_clusters
        self.cluster_counts = torch.zeros(
            n_parts, self.n_clusters, dtype=torch.int, device=device
        )
        self.device = device
        # self.cluster_centers = None
        # self.cluster_counts = None

    def fit(self, centroids, features, part):
        if self.cluster_centers is None or self.cofc_centers is None:
            self.initialize_clusters(features)

        self.cluster_labels = torch.zeros(
            features.size(0), dtype=torch.long, device=self.device
        )
        self.cluster_counts = torch.zeros(
            self.n_clusters, dtype=torch.long, device=self.device
        )

        for i in range(features.size(0)):
            if i < self.n_clusters:
                cluster_id = i
                self.cluster_labels[i] = cluster_id
                self.cluster_counts[cluster_id] += 1
            else:
                point = features[i]
                cofc_distances = self.compute_distance(
                    self.cofc_features, point
                )
                cofc_id = torch.argmin(cofc_distances)

                relevant_centroids_indices = (
                    (self.cofc_cluster_labels == cofc_id).nonzero().squeeze()
                )
                relevant_centroids = self.cluster_centers[
                    relevant_centroids_indices
                ]

                cluster_distances = self.compute_distance(
                    relevant_centroids, point
                )
                relative_cluster_id = torch.argmin(cluster_distances)
                if relevant_centroids_indices.dim() == 0:
                    relevant_centroids_indices = [
                        relevant_centroids_indices.item()
                    ]
                cluster_id = relevant_centroids_indices[relative_cluster_id]
                self.cluster_labels[i] = cluster_id

                self.cluster_counts[cluster_id] += 1
                lr_cluster = 1.0 / self.cluster_counts[cluster_id]
                update_values = lr_cluster * (
                    point - self.cluster_centers[cluster_id]
                )
                self.cluster_centers[cluster_id] += update_values

                lr_cofc = 1.0 / self.cofc_cluster_counts[cofc_id]
                self.cofc_features[cofc_id] += lr_cofc * update_values

        return self.cluster_centers, self.cluster_counts

    def transform(self, features, centroids=None):
        if centroids is not None:
            self.cluster_centers = centroids
        features = features.to(self.device).float()
        self.cluster_centers = self.cluster_centers.to(self.device).float()
        distances = torch.cdist(features, self.cluster_centers)
        nearest_centroid_ids = torch.argmin(distances, dim=1)
        clustered_features = self.cluster_centers[nearest_centroid_ids]
        return clustered_features

    def predict(self, features):
        features = features.to(self.device).float()
        self.cluster_centers = self.cluster_centers.to(self.device).float()
        distances = torch.cdist(features, self.cluster_centers)
        nearest_centroid_ids = torch.argmin(distances, dim=1)
        return nearest_centroid_ids
