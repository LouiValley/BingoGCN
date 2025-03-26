import itertools
import math
import os
import struct

import dgl
import networkx as nx
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
from torch_sparse import sum as sparsesum

from .graph_utils import add_remaining_self_loops, maybe_num_nodes, scatter_add
from .partition_utils import (
    HierarchicalOnlineKMeans,
    OnlineKMeans,
    StreamingKMeans,
    filter_edges,
    get_connected_nodes,
    get_connected_nodes_random_topk,
    reindex_nodes,
)


def read_group_info(file_path):
    with open(file_path, "r") as f:
        node_groups = [int(line.strip()) for line in f]
    return node_groups


def read_centroids(file_path):
    centroids = np.loadtxt(file_path)
    return torch.from_numpy(centroids)


def assign_remaining_nodes(data, train_mask, train_part_labels, n_parts, device):
    remaining_nodes = torch.nonzero(~train_mask, as_tuple=True)[0]
    subgraph_assignments = -torch.ones(data.num_nodes, dtype=torch.long, device=device)
    subgraph_assignments[train_mask] = train_part_labels

    while len(remaining_nodes) > 0:
        node_id = remaining_nodes[0]
        neighbors = data.edge_index[:, data.edge_index[0] == node_id][1]
        subgraph_counts = torch.zeros(n_parts, dtype=torch.long, device=device)
        for neighbor in neighbors:
            if subgraph_assignments[neighbor] >= 0:
                subgraph_counts[subgraph_assignments[neighbor]] += 1
        if subgraph_counts.sum() > 0:
            assigned_subgraph = subgraph_counts.argmax()
            subgraph_assignments[node_id] = assigned_subgraph
        remaining_nodes = remaining_nodes[1:]

    return subgraph_assignments


def find_cut_nodes(data, subgraph_assignments, n_parts):
    cut_nodes = [[] for _ in range(n_parts)]
    for node in range(data.num_nodes):
        neighbors = data.edge_index[:, data.edge_index[0] == node][1]
        for neighbor in neighbors:
            if subgraph_assignments[node] != subgraph_assignments[neighbor]:
                cut_nodes[subgraph_assignments[node]].append(node)
                break
    return cut_nodes


def approximate_cut_nodes(data, cut_nodes, centroids, subgraph_assignments):
    for part, nodes in enumerate(cut_nodes):
        if len(nodes) > 0:
            node_features = data.x[nodes].detach().cpu().numpy()
            kmeans = KMeans(n_clusters=centroids.shape[1], init=centroids[part], n_init=1).fit(
                node_features
            )
            approximated_features = torch.from_numpy(kmeans.cluster_centers_).to(data.x.device)
            data.x[nodes] = approximated_features[kmeans.labels_]

    return data


def evaluate_model_on_partitioned_graph(
    model,
    data,
    sorted_indices,
    sorted_membership,
    n_parts,
    aux_adj,
    ori_membership,
    args,
    g=None,
    ginfeatures=None,
    ginlabels=None,
    ginidx_train=None,
    ginidx_val=None,
    ginidx_test=None,
    dataset=None,
    model_cfg=None,
    centroids=None,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    data = data.to(device)
    sorted_indices = sorted_indices.to(device)
    sorted_membership = sorted_membership.to(device)

    total_val_correct = 0
    total_val_count = 0
    total_test_correct = 0
    total_test_count = 0

    intermediate_outputs = []
    fill_value = 1.0

    if g is not None:
        num_nodes = maybe_num_nodes(data.edge_index, ginfeatures.size(0))
        edge_weight = torch.ones((data.edge_index.size(1),), device=device)
    else:
        if isinstance(data.edge_index, SparseTensor):
            # edge_index = data.edge_index.coo()[:2]
            # num_nodes = maybe_num_nodes(edge_index, data.x.size(0))
            num_nodes = data.num_nodes
            # edge_weight = torch.ones((edge_index[0].size(0),), device=device)
        else:
            num_nodes = maybe_num_nodes(data.edge_index, data.x.size(0))
            edge_weight = torch.ones((data.edge_index.size(1),), device=device)

    if model_cfg == "GCN" or model_cfg == "GAT":
        if isinstance(data.edge_index, SparseTensor):
            dtype = None
            data.edge_index = data.edge_index.fill_value(1.0, dtype=dtype)
            data.edge_index = add_remaining_self_loops(
                data.edge_index, fill_value=fill_value, num_nodes=num_nodes
            )
            deg = sparsesum(data.edge_index, dim=1)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
            edge_weight = torch.ones(
                (data.edge_index.nnz(),),
                dtype=dtype,
                device=device,
            )
            edge_weight = (
                deg_inv_sqrt[data.edge_index.storage.row()]
                * edge_weight
                * deg_inv_sqrt[data.edge_index.storage.col()]
            )
            data.edge_index = data.edge_index.set_value(edge_weight, layout="coo")
            del edge_weight
        else:
            data.edge_index, edge_weight = add_remaining_self_loops(
                data.edge_index, edge_weight, fill_value, num_nodes
            )
            assert edge_weight is not None
            row, col = data.edge_index[0], data.edge_index[1]
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    if g is not None:
        data.val_mask = ginidx_val.to(device)
        data.test_mask = ginidx_test.to(device)
    else:
        data.val_mask = data.val_mask.to(device)
        data.test_mask = data.test_mask.to(device)

    # print(device)
    num_intra_nodes_counts = []
    num_inter_nodes_counts = []
    num_intra_and_inter_nodes_counts = []

    for layer_idx in range(args.num_layers):
        print(f"Layer{layer_idx+1}")
        if g is not None:
            expanded_output = torch.zeros(ginfeatures.size()[0], args.dim_hidden).to(device)
        else:
            expanded_output = torch.zeros(data.num_nodes, args.dim_hidden).to(device)

        if args.global_kmeans and not args.every_X_is_approximated:

            if args.online_kmeans:
                if layer_idx == 0:
                    features = data.x
                else:
                    features = intermediate_outputs[layer_idx - 1]

                kmeans = OnlineKMeans(
                    n_clusters=args.num_kmeans_clusters,
                    device=device,
                    random_state=0,
                    distance_type=args.kmeans_distance,
                )
                cluster_labels, cluster_counts = kmeans.fit(features)
                cluster_centers = kmeans.cluster_centers

                clustered_features = cluster_centers[cluster_labels]
            else:
                # Normal K-Means
                if layer_idx == 0:
                    features = data.x
                else:
                    features = intermediate_outputs[layer_idx - 1]

                kmeans = KMeans(
                    n_clusters=args.num_kmeans_clusters,
                    random_state=0,
                    n_init=10,
                    init="random",
                ).fit(features.detach().cpu().numpy())
                cluster_labels = torch.from_numpy(kmeans.labels_).to(device).long()
                cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(device)

            if args.kmeans_approximation == "center":
                clustered_features = cluster_centers[cluster_labels]
            elif args.kmeans_approximation == "average":
                clustered_features = torch.zeros_like(features)
                for cluster_id in range(args.num_kmeans_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_features = features[cluster_mask]
                    cluster_mean = cluster_features.mean(dim=0)
                    clustered_features[cluster_mask] = cluster_mean

        if args.outgoing_kmeans:
            part_outgoing_centroids_features = []
            part_outgoing_centroids_labels = []
            part_outgoing_nodes_index = []
            if layer_idx == 0:
                features = data.x
            else:
                features = intermediate_outputs[layer_idx - 1]

            cofc_features_list = []
            L1kid_L2kid_list = []

            if isinstance(data.edge_index, SparseTensor):
                row, col, _ = data.edge_index.coo()
            else:
                row, col = data.edge_index

            for part in range(n_parts):
                part_mask = sorted_membership == part
                part_indices = sorted_indices[part_mask]
                part_indices_tensor = part_indices.to(device)
                if isinstance(data.edge_index, SparseTensor):
                    # SparseTensorからCOO形式のインデックスを取得

                    # 取得したインデックスを使用して処理を行う
                    outgoing_nodes_mask = (
                        torch.isin(row, part_indices_tensor) & ~torch.isin(col, part_indices_tensor)
                    ) | (
                        torch.isin(col, part_indices_tensor) & ~torch.isin(row, part_indices_tensor)
                    )
                    outgoing_nodes = torch.cat(
                        [
                            row[outgoing_nodes_mask],
                            col[outgoing_nodes_mask],
                        ]
                    ).unique()
                else:
                    outgoing_nodes_mask = (
                        torch.isin(data.edge_index[0], part_indices_tensor)
                        & ~torch.isin(data.edge_index[1], part_indices_tensor)
                    ) | (
                        torch.isin(data.edge_index[1], part_indices_tensor)
                        & ~torch.isin(data.edge_index[0], part_indices_tensor)
                    )
                    outgoing_nodes = torch.cat(
                        [
                            data.edge_index[0, outgoing_nodes_mask],
                            data.edge_index[1, outgoing_nodes_mask],
                        ]
                    ).unique()
                outgoing_nodes = outgoing_nodes[torch.isin(outgoing_nodes, part_indices_tensor)]
                outgoing_features = features[outgoing_nodes]

                # print(len(outgoing_nodes))

                if args.fixed_centroid_ratio:
                    num_kmeans_clusters = math.ceil(len(outgoing_features) * args.centroid_ratio)
                else:
                    num_kmeans_clusters = args.num_kmeans_clusters

                if layer_idx == 0:
                    if args.online_kmeans:
                        kmeans = OnlineKMeans(
                            n_clusters=num_kmeans_clusters,
                            device=device,
                            random_state=0,
                            distance_type=args.kmeans_distance,
                        )
                        outgoing_cluster_labels, cluster_counts = kmeans.fit(outgoing_features)
                        outgoing_cluster_centers = kmeans.cluster_centers
                    else:
                        kmeans = KMeans(
                            n_clusters=num_kmeans_clusters,
                            random_state=0,
                            n_init=10,
                            init="random",
                        ).fit(outgoing_features.detach().cpu().numpy())

                        outgoing_cluster_labels = torch.from_numpy(kmeans.labels_).to(device).long()
                        outgoing_cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(
                            device
                        )

                else:
                    if args.hierarchical_kmeans:
                        num_cofc = num_kmeans_clusters // args.size_l1_cent_group
                        if num_cofc == 0:
                            num_cofc = 1

                        kmeans = HierarchicalOnlineKMeans(
                            n_clusters=num_kmeans_clusters,
                            n_cofc=num_cofc,
                            device=device,
                            random_state=0,
                            distance_type=args.kmeans_distance,
                            layer=layer_idx,
                        )
                        (
                            outgoing_cluster_labels,
                            cofc_features,
                            cofc_cluster_labels,
                        ) = kmeans.fit(outgoing_features)
                        outgoing_cluster_centers = kmeans.cluster_centers
                        cofc_features_list.append(cofc_features)
                    else:
                        if args.online_kmeans:
                            kmeans = OnlineKMeans(
                                n_clusters=num_kmeans_clusters,
                                device=device,
                                random_state=0,
                                distance_type=args.kmeans_distance,
                            )
                            outgoing_cluster_labels, cluster_counts = kmeans.fit(outgoing_features)
                            outgoing_cluster_centers = kmeans.cluster_centers
                        else:
                            kmeans = KMeans(
                                n_clusters=num_kmeans_clusters,
                                random_state=0,
                                n_init=10,
                                init="random",
                            ).fit(outgoing_features.detach().cpu().numpy())

                            outgoing_cluster_labels = (
                                torch.from_numpy(kmeans.labels_).to(device).long()
                            )
                            outgoing_cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(
                                device
                            )
                    # clustered_features = cluster_centers[cluster_labels]

                    # outgoing_cluster_labels = cluster_labels
                    # outgoing_cluster_centers = cluster_centers

                # outgoing_clustered_features = outgoing_cluster_centers[
                #     outgoing_cluster_labels
                # ]
                # # Place updated features back into the full feature set
                # features[outgoing_nodes_indices] = outgoing_clustered_features

                part_outgoing_centroids_features.append(outgoing_cluster_centers)

                if args.flowgnn_debug:
                    os.makedirs(
                        f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part}",
                        exist_ok=True,
                    )
                    np.savetxt(
                        f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part}/l{layer_idx+1}_p{part}_L2kinfo.txt",
                        outgoing_cluster_centers.detach().cpu().numpy(),
                        delimiter=",",
                    )

                part_outgoing_centroids_labels.append(outgoing_cluster_labels)
                part_outgoing_nodes_index.append(outgoing_nodes)

            if args.flowgnn_debug and layer_idx != 0 and args.hierarchical_kmeans:
                cofc_features_tensor = torch.cat(cofc_features_list)

                cofc_features_tensor = cofc_features_tensor.detach().cpu().numpy().astype(np.int32)

                os.makedirs(f"./pretrained_model/Kmeans/l{layer_idx+1}", exist_ok=True)
                # # テキストファイルに保存
                # with open(
                #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L1kinfo.txt",
                #     "w",
                # ) as f:
                #     for item in cofc_features_tensor:
                #         f.write(f"{item.item()}\n")
                np.savetxt(
                    f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L1kinfo.txt",
                    cofc_features_tensor,
                    delimiter=",",
                )
                # バイナリファイルとして保存
                # torch.save(
                #     cofc_features_tensor,
                #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L1kinfo.bin",
                # )
                with open(
                    f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L1kinfo.bin",
                    "wb",
                ) as f:
                    f.write(cofc_features_tensor.tobytes())
                # # ファイルに書き出すためにインデックスを抽出します。
                # with open(
                #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L1kid_L2kid.txt",
                #     "w",
                # ) as f:
                for i in range(args.num_kmeans_clusters // 8):
                    indices = (cofc_cluster_labels == i).nonzero(as_tuple=True)[0]
                    indices_str = ", ".join(map(str, indices.tolist()))
                    # f.write(indices_str + "\n")

                    L1kid_L2kid_list.append(indices_str)

        if args.flowgnn_debug:

            os.makedirs(f"./pretrained_model/Kmeans/l{layer_idx+1}", exist_ok=True)

            def to_flat_list(tensors):
                # テンソルを1次元リストに完全に平滑化
                return list(
                    itertools.chain.from_iterable(
                        t.cpu().detach().numpy().flatten().tolist()  # .flatten()を追加して1次元化
                        for t in tensors
                    )
                )

            all_counts = []

            for tensor in part_outgoing_centroids_labels:
                # GPUからCPUへテンソルを移動し、NumPy配列に変換
                tensor = tensor.cpu().numpy()
                # テンソル内のクラスタラベルの出現回数をカウント
                counts = np.bincount(tensor, minlength=args.num_kmeans_clusters)
                all_counts.append(counts)

            # 縦方向に結合
            combined_counts = np.vstack(all_counts)

            # 結合した配列を1列のベクトルに平滑化
            flattened_counts = combined_counts.flatten().astype(np.int32)

            # テキストファイルとして保存
            np.savetxt(
                f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_local.txt",
                flattened_counts,
                fmt="%d",
            )

            # バイナリファイルとして保存
            # flattened_counts.tofile(
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_local.bin"
            # )
            with open(
                f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_local.bin",
                "wb",
            ) as f:
                f.write(flattened_counts.tobytes())

            # データの取得と変換
            # flat_features = to_flat_list(part_outgoing_centroids_features)
            flat_labels = to_flat_list(part_outgoing_centroids_labels)
            # flat_indexes = to_flat_list(part_outgoing_nodes_index)

            # ファイルに保存
            # with open(
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_outgoing_kinfo_flattened.txt",
            #     "w",
            # ) as file:
            #     for feature in flat_features:
            #         file.write(
            #             str(feature) + "\n"
            #         )  # 各特徴を新しい行に書き込む
            cpu_part_outgoing_centroids_features = [
                tensor.detach().cpu().numpy() for tensor in part_outgoing_centroids_features
            ]

            concatenated_features = np.concatenate(cpu_part_outgoing_centroids_features, axis=0)

            np.savetxt(
                f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L2kinfo.txt",
                concatenated_features,
                delimiter=",",
            )
            # with open(f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L2kinfo.bin", "wb") as f:
            #     f.write(concatenated_features.tobytes())

            with open(
                f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_outgoing_L2kid.txt",
                "w",
            ) as file:
                for label in flat_labels:
                    file.write(str(label) + "\n")

            np.savetxt(
                f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_L1kid_L2kid.txt",
                L1kid_L2kid_list,
                fmt="%s",  # 整数として保存
                delimiter=",",  # カンマ区切りで保存
            )

            # with open(
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/outgoing_nodes_index_layer{layer_idx}.txt",
            #     "w",
            # ) as file:
            #     for index in flat_indexes:
            #         file.write(
            #             str(index) + "\n"
            #         )  # 各インデックスを新しい行に書き込む

        total_inter = 0
        total_intra = 0
        total_outgoing = 0

        num_outgoing_nodes_list = list()
        num_inter_nodes_list = list()

        node_part_list = []
        incoming_qid_list = []
        outgoing_qid_list = []

        for part in range(n_parts):
            part_node_part_list = []
            part_incoming_qid_list = []
            # outgoing_qid_list = []

            num_intra_nodes_counts = []
            num_inter_nodes_counts = []
            num_intra_and_inter_nodes_counts = []

            # print(f"layer_idx: {layer_idx}, part: {part}")
            part_mask = sorted_membership == part
            part_indices = sorted_indices[part_mask]

            if args.flowgnn_debug and layer_idx != 0:
                os.makedirs(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}",
                    exist_ok=True,
                )
                inner_and_outgoing_x = expanded_output[part_indices].detach().cpu().numpy()
                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_inner_and_outgoing_x.bin",
                    "wb",
                ) as f:
                    f.write(inner_and_outgoing_x.tobytes())
                np.savetxt(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_inner_and_outgoing_x.txt",
                    inner_and_outgoing_x,
                    # fmt="%d",
                    # delimiter=",",
                )

            if args.inter_cluster is True or args.no_inter_cluster:
                if args.random_sampling or args.topk_sampling:
                    (
                        new_part_indices,
                        additional_nodes_indices,
                        num_intra_nodes,
                        num_inter_nodes,
                        num_intra_and_inter_nodes,
                        num_outgoing_nodes,
                    ) = get_connected_nodes_random_topk(
                        part_indices,
                        data.edge_index,
                        args.inter_sparsity,
                        args,
                    )
                else:
                    (
                        new_part_indices,
                        additional_nodes_indices,
                        num_intra_nodes,
                        num_inter_nodes,
                        num_intra_and_inter_nodes,
                        num_outgoing_nodes,
                    ) = get_connected_nodes(
                        part_indices,
                        data.edge_index,
                        args.inter_sparsity,
                        args,
                    )
                if args.outgoing_kmeans:
                    if layer_idx == 0:
                        additional_features = data.x[additional_nodes_indices]
                        part_data_x = data.x[new_part_indices]
                    else:
                        part_data_x = intermediate_outputs[layer_idx - 1][new_part_indices]
                        additional_features = intermediate_outputs[layer_idx - 1][
                            additional_nodes_indices
                        ]

                    additional_nodes_mask = torch.isin(
                        new_part_indices,
                        additional_nodes_indices.clone().detach().to(device),
                    )

                    if args.flowgnn_debug:
                        outgoing_qid_list.append(additional_nodes_indices)

                    for idx, node_idx in enumerate(additional_nodes_indices):
                        node_part = ori_membership[node_idx]
                        if args.flowgnn_debug:
                            node_part_list.append(node_part)
                            part_node_part_list.append(node_part)
                        cluster_labels = part_outgoing_centroids_labels[node_part].cuda()
                        cluster_centers = part_outgoing_centroids_features[node_part].cuda()
                        outgoing_nodes_index = part_outgoing_nodes_index[node_part].cuda()

                        cluster_idx = cluster_labels[
                            torch.where(outgoing_nodes_index == node_idx)[0]
                        ]

                        if args.flowgnn_debug:
                            part_incoming_qid_list.append(
                                torch.where(outgoing_nodes_index == node_idx)[0]
                            )
                            incoming_qid_list.append(
                                torch.where(outgoing_nodes_index == node_idx)[0]
                            )

                        # additional_nodes_mask が True のインデックスを取得
                        true_indices = torch.where(additional_nodes_mask)[0]

                        # true_indices から実際に更新したいインデックスを取得
                        update_index = true_indices[idx]

                        # 元のテンソルを直接更新
                        part_data_x[update_index] = cluster_centers[cluster_idx]

                    # num_inter_nodes = args.num_kmeans_clusters
                    num_intra_and_inter_nodes = num_intra_nodes + num_inter_nodes

                elif args.partial_kmeans:
                    if layer_idx == 0:
                        additional_features = data.x[additional_nodes_indices]
                    else:
                        additional_features = intermediate_outputs[layer_idx - 1][
                            additional_nodes_indices
                        ]

                    if (
                        len(additional_nodes_indices) > 0
                        and additional_features.size(0) >= args.num_kmeans_clusters
                    ):
                        if args.online_kmeans:
                            kmeans = OnlineKMeans(
                                n_clusters=args.num_kmeans_clusters,
                                device=device,
                                random_state=0,
                                distance_type=args.kmeans_distance,
                            )
                            cluster_labels, cluster_counts = kmeans.fit(additional_features)
                            cluster_centers = kmeans.cluster_centers

                        elif args.training_centroids:
                            if layer_idx == 0:
                                kmeans = StreamingKMeans(args.num_kmeans_clusters, n_parts, device)
                                kmeans.cluster_centers = centroids[0]
                                cluster_centers = kmeans.cluster_centers
                                cluster_labels = kmeans.predict(additional_features)

                                # kmeans = StreamingKMeans(args.num_kmeans_clusters, device)
                                # centroids = np.loadtxt(f"layer{layer_idx}_centroids/partition{part}_centroids.txt")
                                # centroids_all = torch.from_numpy(centroids).to(device)
                                # cluster_centers = centroids_all
                                # kmeans.cluster_centers = cluster_centers
                                # cluster_labels = kmeans.predict(additional_features)

                            else:
                                kmeans = OnlineKMeans(
                                    n_clusters=args.num_kmeans_clusters,
                                    device=device,
                                    random_state=0,
                                    distance_type=args.kmeans_distance,
                                )
                                # centroids_all = [None] * args.n_parts

                                # centroids = np.loadtxt(f"layer{layer_idx}_centroids/partition{part}_centroids.txt")
                                # centroids_all = torch.from_numpy(centroids).to(device)

                                # cluster_centers = centroids_all
                                # kmeans.cluster_centers = cluster_centers
                                # kmeans.initialize_centers

                                cluster_labels, cluster_counts = kmeans.fit(
                                    additional_features,
                                )

                                # np.savetxt(f"./cluster_counts_l{layer_idx+1}_p{part+1}.txt", cluster_counts.cpu().numpy())

                                cluster_centers = kmeans.cluster_centers

                                # kmeans = StreamingKMeans(args.num_kmeans_clusters, device)
                                # centroids = np.loadtxt(f"layer{layer_idx}_centroids/partition{part}_centroids.txt")
                                # centroids_all = torch.from_numpy(centroids).to(device)
                                # cluster_centers = centroids_all
                                # kmeans.cluster_centers = cluster_centers
                                # cluster_labels = kmeans.predict(additional_features)

                        else:
                            kmeans = KMeans(
                                n_clusters=args.num_kmeans_clusters,
                                random_state=0,
                                n_init=10,
                                init="random",
                            ).fit(additional_features.detach().cpu().numpy())

                            cluster_labels = torch.from_numpy(kmeans.labels_).to(device).long()
                            cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(device)

                        # # cluster_centersをファイルに保存
                        # centroids_save_path = (
                        #     f"layer{layer_idx}_centroids/partition{part}_centroids.txt"
                        # )
                        # np.savetxt(centroids_save_path, cluster_centers.detach().cpu().numpy())

                        if not args.no_kmeans_approximation:
                            if args.kmeans_approximation == "center":
                                clustered_features = cluster_centers[cluster_labels].float()
                                # print(cluster_centers)
                            elif args.kmeans_approximation == "average":
                                clustered_features = torch.zeros_like(
                                    additional_features, dtype=torch.float
                                )
                                for cluster_id in range(args.num_kmeans_clusters):
                                    cluster_mask = cluster_labels == cluster_id
                                    cluster_features = additional_features[cluster_mask]
                                    cluster_mean = cluster_features.mean(dim=0)
                                    clustered_features[cluster_mask] = cluster_mean

                    else:
                        print(
                            f"Skipping KMeans clustering for layer {layer_idx}, part {part} due to insufficient samples."
                        )  # noqa
                        clustered_features = additional_features

                    if layer_idx == 0:
                        part_data_x = data.x[new_part_indices]
                    else:
                        part_data_x = intermediate_outputs[layer_idx - 1][new_part_indices]

                    additional_nodes_mask = torch.isin(
                        new_part_indices,
                        additional_nodes_indices.clone().detach().to(device),
                    )

                    if args.no_kmeans_approximation is False:
                        part_data_x[additional_nodes_mask] = clustered_features
                    elif args.no_kmeans_approximation is True:
                        if layer_idx == 0:
                            part_data_x[additional_nodes_mask] = data.x[additional_nodes_indices]
                        else:
                            part_data_x[additional_nodes_mask] = intermediate_outputs[
                                layer_idx - 1
                            ][additional_nodes_indices]

                    num_inter_nodes = args.num_kmeans_clusters
                    num_intra_and_inter_nodes = num_intra_nodes + num_inter_nodes

                elif args.global_kmeans and not args.every_X_is_approximated:
                    if layer_idx == 0:
                        part_data_x = data.x[new_part_indices]
                    else:
                        part_data_x = intermediate_outputs[layer_idx - 1][new_part_indices]

                    # Create a mask for additional nodes in the current partition
                    additional_nodes_mask = torch.zeros(data.x.size(0), dtype=torch.bool).to(device)
                    additional_nodes_mask[additional_nodes_indices] = True
                    additional_nodes_mask = additional_nodes_mask[new_part_indices]

                    # Replace features of additional nodes with clustered features
                    part_data_x[additional_nodes_mask] = clustered_features[
                        additional_nodes_indices
                    ]

                    num_inter_nodes = args.num_kmeans_clusters
                    num_intra_and_inter_nodes = num_intra_nodes + num_inter_nodes

                else:
                    if args.every_X_is_approximated:
                        if args.global_kmeans:
                            if layer_idx == 0:
                                if args.online_kmeans:
                                    # part_data_xのすべての行に対してOnlineKMeansを適用
                                    kmeans = OnlineKMeans(
                                        n_clusters=args.num_kmeans_clusters,
                                        device=device,
                                        random_state=0,
                                        distance_type=args.kmeans_distance,
                                    )
                                    cluster_labels, cluster_counts = kmeans.fit(data.x)
                                    cluster_centers = kmeans.cluster_centers
                                else:
                                    # part_data_xのすべての行に対してKMeansを適用
                                    kmeans = KMeans(
                                        n_clusters=args.num_kmeans_clusters,
                                        random_state=0,
                                        n_init=10,
                                        init="random",
                                    ).fit(data.x.detach().cpu().numpy())
                                    cluster_labels = (
                                        torch.from_numpy(kmeans.labels_).to(device).long()
                                    )
                                    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(
                                        device
                                    )
                                data.x = cluster_centers[cluster_labels]
                                part_data_x = data.x[new_part_indices]
                            else:
                                if args.online_kmeans:
                                    # part_data_xのすべての行に対してOnlineKMeansを適用
                                    kmeans = OnlineKMeans(
                                        n_clusters=args.num_kmeans_clusters,
                                        device=device,
                                        random_state=0,
                                        distance_type=args.kmeans_distance,
                                    )
                                    cluster_labels, cluster_counts = kmeans.fit(
                                        intermediate_outputs
                                    )
                                    cluster_centers = kmeans.cluster_centers
                                else:
                                    # part_data_xのすべての行に対してKMeansを適用
                                    kmeans = KMeans(
                                        n_clusters=args.num_kmeans_clusters,
                                        random_state=0,
                                        n_init=10,
                                        init="random",
                                    ).fit(intermediate_outputs.detach().cpu().numpy())
                                    cluster_labels = (
                                        torch.from_numpy(kmeans.labels_).to(device).long()
                                    )
                                    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(
                                        device
                                    )
                                intermediate_outputs = cluster_centers[cluster_labels]
                                part_data_x = intermediate_outputs[layer_idx - 1][new_part_indices]
                        else:
                            if layer_idx == 0:
                                part_data_x = data.x[new_part_indices]
                            else:
                                part_data_x = intermediate_outputs[layer_idx - 1][new_part_indices]

                            if args.online_kmeans:
                                # part_data_xのすべての行に対してOnlineKMeansを適用
                                kmeans = OnlineKMeans(
                                    n_clusters=args.num_kmeans_clusters,
                                    device=device,
                                    random_state=0,
                                    distance_type=args.kmeans_distance,
                                )
                                cluster_labels, cluster_counts = kmeans.fit(part_data_x)
                                cluster_centers = kmeans.cluster_centers
                            else:
                                # part_data_xのすべての行に対してKMeansを適用
                                kmeans = KMeans(
                                    n_clusters=args.num_kmeans_clusters,
                                    random_state=0,
                                    n_init=10,
                                    init="random",
                                ).fit(part_data_x.detach().cpu().numpy())
                                cluster_labels = torch.from_numpy(kmeans.labels_).to(device).long()
                                cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(
                                    device
                                )

                            part_data_x = cluster_centers[cluster_labels]
                    else:
                        if layer_idx == 0:
                            part_data_x = data.x[new_part_indices]
                        else:
                            part_data_x = intermediate_outputs[layer_idx - 1][new_part_indices]

                if isinstance(data.edge_index, SparseTensor):
                    part_edge_index, part_edge_weight = filter_edges(
                        data.edge_index, new_part_indices
                    )
                else:
                    part_edge_index, part_edge_weight = filter_edges(
                        data.edge_index, new_part_indices, edge_weight
                    )
                part_edge_index, _ = reindex_nodes(part_edge_index, new_part_indices)

                if args.flowgnn_debug:
                    part_edge_weight = None

                if args.save_partition_bin:
                    edge_index, _ = remove_self_loops(part_edge_index)

                    # Save part_data_x to txt file
                    with open("./gnn_ticket/p1_x.txt", "w") as file:
                        for i in range(part_data_x.shape[0]):
                            values = [str(val.item()) for val in part_data_x[i]]
                            file.write(" ".join(values) + "\n")

                    # Save part_data_x to binary file
                    with open("./gnn_ticket/p1_x.bin", "wb") as file:
                        for i in range(part_data_x.shape[0]):
                            for j in range(part_data_x.shape[1]):
                                file.write(struct.pack("f", part_data_x[i][j].item()))

                    # Save edge_index to txt file
                    edge_index = edge_index.cpu()
                    with open("./gnn_ticket/p1_edge_index.txt", "w") as file:
                        file.write("[[")
                        for i in range(edge_index.shape[1]):
                            file.write(str(edge_index[0][i].item()) + ", ")
                        file.write("],\n")

                        file.write(" [")
                        for i in range(edge_index.shape[1]):
                            file.write(str(edge_index[1][i].item()) + ", ")
                        file.write("]]\n")

                    # Save edge_index to binary file
                    with open("./gnn_ticket/p1_edge_index.bin", "wb") as file:
                        for i in range(edge_index.shape[1]):
                            file.write(
                                struct.pack(
                                    "ii",
                                    edge_index[0][i].item(),
                                    edge_index[1][i].item(),
                                )
                            )

                all_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
                all_nodes_mask[part_indices] = True
                enhanced_all_nodes_mask = torch.zeros(
                    data.num_nodes, dtype=torch.bool, device=device
                )
                enhanced_all_nodes_mask[new_part_indices] = True

                part_val_mask = data.val_mask[enhanced_all_nodes_mask]
                part_test_mask = data.test_mask[enhanced_all_nodes_mask]

                added_nodes_indices = torch.where(enhanced_all_nodes_mask & ~all_nodes_mask)[0]
                part_val_mask = data.val_mask[enhanced_all_nodes_mask].clone()
                enhanced_indices = torch.where(enhanced_all_nodes_mask)[0]
                new_indices_for_added_nodes = torch.searchsorted(
                    enhanced_indices, added_nodes_indices
                )
                part_val_mask[new_indices_for_added_nodes] = False

                added_nodes_indices = torch.where(enhanced_all_nodes_mask & ~all_nodes_mask)[0]
                part_test_mask = data.test_mask[enhanced_all_nodes_mask].clone()
                enhanced_indices = torch.where(enhanced_all_nodes_mask)[0]
                new_indices_for_added_nodes = torch.searchsorted(
                    enhanced_indices, added_nodes_indices
                )
                part_test_mask[new_indices_for_added_nodes] = False

                num_intra_nodes_counts.append(num_intra_nodes)
                num_inter_nodes_counts.append(num_inter_nodes)
                num_intra_and_inter_nodes_counts.append(num_intra_and_inter_nodes)

                if args.save_partition_bin and part == 0 and layer_idx == 0:
                    # organize edge_list
                    bin_edge_index = part_edge_index.cpu()
                    bin_edges = [{"u": int(u), "v": int(v)} for u, v in bin_edge_index.t().numpy()]
                    # to bin file
                    with open("cora_g1_edge_list.bin", "wb") as f:
                        for edge in bin_edges:
                            #  u->v
                            f.write(struct.pack("ii", edge["u"], edge["v"]))

                    x = part_data_x.cpu().numpy()
                    with open("cora_g1_node_feature.bin", "wb") as f:
                        f.write(x.tobytes())

            # if layer_idx == 0:
            #     print(
            #         f"Partition{part}: Inner nodes = {num_intra_nodes-num_outgoing_nodes}, Outgoing nodes = {num_outgoing_nodes}, Incoming nodes = {num_inter_nodes}"
            #     )

            num_outgoing_nodes_list.append(num_outgoing_nodes)
            num_inter_nodes_list.append(num_inter_nodes)

            total_inter += num_inter_nodes
            total_outgoing += num_outgoing_nodes
            total_intra += num_intra_nodes

            if args.flowgnn_debug:
                # os.makedirs(
                #     f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}",
                #     exist_ok=True,
                # )

                # part_node_part_list
                # part_incoming_qid_list
                os.makedirs(
                    f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}",
                    exist_ok=True,
                )
                # node_part_tensor = torch.tensor(part_node_part_list)

                # テキストファイルとして保存
                with open(
                    f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}/l{layer_idx+1}_p{part+1}_incoming_outgoing_pid.txt",
                    "w",
                ) as f:
                    for item in part_node_part_list:
                        f.write(f"{item}\n")

                # torch.save(
                #     node_part_tensor.numpy.astype(np.int32),
                #     f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}/l{layer_idx+1}_p{part+1}_incoming_outgoing_pid.bin",
                # )
                part_node_part_list = np.array(part_node_part_list).astype(np.int32)

                with open(
                    f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}/l{layer_idx+1}_p{part+1}_incoming_outgoing_pid.bin",
                    "wb",
                ) as f:
                    f.write(part_node_part_list.tobytes())

                incoming_qid_tensor = torch.tensor(part_incoming_qid_list)

                with open(
                    f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}/l{layer_idx+1}_p{part+1}_incoming_outgoing_qid.txt",
                    "w",
                ) as f:
                    for item in incoming_qid_tensor:
                        f.write(f"{item}\n")

                # # バイナリファイルとして保存
                # torch.save(
                #     incoming_qid_tensor,
                #     f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}/l{layer_idx+1}_p{part+1}_incoming_outgoing_qid.bin",
                # )

                incoming_qid_tensor = incoming_qid_tensor.numpy().astype(np.int32)
                with open(
                    f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}/l{layer_idx+1}_p{part+1}_incoming_outgoing_qid.bin",
                    "wb",
                ) as f:
                    f.write(incoming_qid_tensor.tobytes())

                csr_sparse_tensor = part_data_x.to_sparse()
                indices_s = csr_sparse_tensor.indices().cpu().numpy()
                values = csr_sparse_tensor.values().detach().cpu().numpy()
                csr_matrix = scipy.sparse.csr_matrix(
                    (values, (indices_s[0], indices_s[1])),
                    shape=csr_sparse_tensor.shape,
                )
                csr_data = csr_matrix.data
                csr_indices = csr_matrix.indices
                csr_indptr = csr_matrix.indptr
                combined_csr_data_indices = np.row_stack((csr_indices, csr_data))
                print("Data & csr_indices:", combined_csr_data_indices.shape)
                print("csr_indptr len:", len(csr_indptr.data))
                os.makedirs(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}",
                    exist_ok=True,
                )

                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_csr_value.bin",
                    "wb",
                ) as f:
                    f.write(csr_data.tobytes())

                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_csr_indice.bin",
                    "wb",
                ) as f:
                    f.write(csr_indices.tobytes())

                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_csr_indptr.bin",
                    "wb",
                ) as f:
                    f.write(csr_indptr.tobytes())

                flattened_csr_value = csr_data.ravel(order="F")
                flattened_csr_indices = csr_indices.ravel(order="F")
                flattened_csr_indptr = csr_indptr.ravel(order="F")
                np.savetxt(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_csr_value_flatten.txt",
                    flattened_csr_value,
                    fmt="%f",
                    delimiter=",",
                )
                np.savetxt(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_csr_indices_flatten.txt",
                    flattened_csr_indices,
                    fmt="%d",
                    delimiter=",",
                )
                np.savetxt(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_csr_indptr_flatten.txt",
                    flattened_csr_indptr,
                    fmt="%d",
                    delimiter=",",
                )

                # with open(
                #     f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}_csr_value.log", "w"
                # ) as f:
                #     for i in range(len(csr_data)):
                #         f.write(f"{csr_data[i]}\n")

                # with open(
                #     f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}_csr_indices.log", "w"
                # ) as f:
                #     for i in range(len(csr_indices)):
                #         f.write(f"{csr_indices[i]}\n")

                # with open(
                #     f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}_csr_indptr.log", "w"
                # ) as f:
                #     for i in range(len(csr_indptr)):
                #         f.write(f"{csr_indptr[i]}\n")

                # organize edge_list
                bin_edge_index = part_edge_index.cpu()
                bin_edges = [{"u": int(u), "v": int(v)} for u, v in bin_edge_index.t().numpy()]
                # os.makedirs(
                #     f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}", exist_ok=True
                # )
                # バイナリファイルとしてエッジを保存
                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_edges.bin",
                    "wb",
                ) as f:
                    for edge in bin_edges:
                        f.write(struct.pack("ii", edge["u"], edge["v"]))

                # テキストファイルとしてエッジを保存
                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_edges.txt",
                    "w",
                ) as f:
                    for edge in bin_edges:
                        f.write(f"{edge['u']} {edge['v']}\n")

                # バイナリファイルとしてフィーチャーを保存
                x = part_data_x.detach().cpu().numpy()
                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_features.bin",
                    "wb",
                ) as f:
                    f.write(x.tobytes())

                # テキストファイルとしてフィーチャーを保存
                with open(
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_features.txt",
                    "w",
                ) as f:
                    # numpy 配列の各行をテキストファイルに書き出す
                    for feature in x:
                        f.write(" ".join(map(str, feature)) + "\n")

                filename = (
                    f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_input_info.txt"
                )

                # Collect the necessary information
                num_nodes = part_data_x.size(0)
                num_edges = part_edge_index.size(1)
                num_csrdata = len(csr_data)
                num_csrptr = len(csr_indptr)
                num_outgoing = num_outgoing_nodes
                num_incoming = num_inter_nodes

                # Write the information to the file
                with open(filename, "w") as file:
                    file.write(f"{num_nodes}\n")
                    file.write(f"{num_edges}\n")
                    file.write(f"{num_csrdata}\n")
                    file.write(f"{num_csrptr}\n")
                    file.write(f"{part}\n")
                    file.write(f"{num_outgoing}\n")
                    file.write(f"{num_incoming}\n")

            # if (
            #     args.dataset == "Cora"
            #     or args.dataset == "ogbn-arxiv"
            #     or args.dataset == "Reddit"
            # ):
            #     num_part_nodes = len(part_data_x)
            #     row, col = part_edge_index
            #     part_edge_index = SparseTensor(
            #         row=row,
            #         col=col,
            #         sparse_sizes=(num_part_nodes, num_part_nodes),
            #     )

            if model_cfg == "GCN" or model_cfg == "GAT":
                if args.original_edge_weight:
                    model.eval()
                    model = model.to(device)
                    logits = model(
                        part_data_x.to(device),
                        part_edge_index.to(device),
                        return_intermediate=True,
                        layer_idx=layer_idx,
                    )
                else:
                    model.eval()
                    if args.kmeans_before_relu and layer_idx != 0:
                        part_data_x = F.relu(part_data_x)
                    # if layer_idx == 0:
                    #     os.makedirs(
                    #         f"./pretrained_model/Kmeans/l{layer_idx+1}/p{part+1}",
                    #         exist_ok=True,
                    #     )
                    #     csr_sparse_tensor = part_data_x.to_sparse()
                    #     indices_s = csr_sparse_tensor.indices().cpu().numpy()
                    #     values = csr_sparse_tensor.values().detach().cpu().numpy()
                    #     csr_matrix = scipy.sparse.csr_matrix(
                    #         (values, (indices_s[0], indices_s[1])),
                    #         shape=csr_sparse_tensor.shape,
                    #     )
                    #     csr_data = csr_matrix.data
                    #     csr_indptr = csr_matrix.indptr
                    #     os.makedirs(
                    #         f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}",
                    #         exist_ok=True,
                    #     )
                    #     x = part_data_x.detach().cpu().numpy()
                    #     filename = f"./pretrained_model/Input/l{layer_idx+1}/p{part+1}/p{part+1}_input_info.txt"
                    #     num_nodes = part_data_x.size(0)
                    #     # sparse_tensor = torch_sparse.SparseTensor(row=row, col=col, value=value)
                    #     if isinstance(part_edge_index, torch.Tensor):
                    #         num_edges = part_edge_index.size(-1)
                    #     else:
                    #         num_edges = part_edge_index.nnz()
                    #     num_csrdata = len(csr_data)
                    #     num_csrptr = len(csr_indptr)
                    #     num_outgoing = num_outgoing_nodes
                    #     num_incoming = num_inter_nodes
                    #     with open(filename, "w") as file:
                    #         file.write(f"{num_nodes}\n")
                    #         file.write(f"{num_edges}\n")
                    #         file.write(f"{num_csrdata}\n")
                    #         file.write(f"{num_csrptr}\n")
                    #         file.write(f"{part}\n")
                    #         file.write(f"{num_outgoing}\n")
                    #         file.write(f"{num_incoming}\n")

                    if isinstance(part_edge_index, SparseTensor):
                        model = model.to(device)
                        part_data_x = part_data_x.to(device)
                        part_edge_index = part_edge_index.to(device)
                        logits = model(
                            part_data_x,
                            part_edge_index,
                            return_intermediate=True,
                            layer_idx=layer_idx,
                            # edge_weight=part_edge_weight,
                            edge_weight=None,
                            part=part,
                        )
                        # logits = logits.cpu()
                        # logits = logits.to(device)
                    else:
                        model = model.to(device)
                        logits = model(
                            part_data_x.to(device),
                            part_edge_index.to(device),
                            return_intermediate=True,
                            layer_idx=layer_idx,
                            edge_weight=part_edge_weight.to(device),
                            part=part,
                        )
                        # logits = logits.to(device)

            elif model_cfg == "GIN":
                node_num = len(new_part_indices)
                sub_g = dgl.graph(
                    (part_edge_index[0], part_edge_index[1]),
                    num_nodes=node_num,
                ).to(device)
                local_part_edge_index, _ = reindex_nodes(part_edge_index, new_part_indices)
                adj = nx.adjacency_matrix(nx.from_edgelist(local_part_edge_index.t().cpu().numpy()))
                adj = torch.from_numpy(adj.todense()).to(device).float()
                adj.fill_diagonal_(0)
                sub_g.adjacency_matrix = adj
                # adjacency_matrix_list = sub_g.adjacency_matrix.tolist()
                # with open("g_sub_adjacency_matrix.txt", "w") as file:
                #     for row in adjacency_matrix_list:
                #         row_str = " ".join(str(val) for val in row)
                #         file.write(row_str + "\n")
                logits = model(
                    sub_g,
                    part_data_x,
                    0,
                    0,
                    # sparsity=args.sparsity_list,
                    return_intermediate=True,
                    layer_idx=layer_idx,
                    edge_weight=part_edge_weight,
                    part_idx=part,
                )

            # if (
            #     args.dataset == "Cora"
            #     or args.dataset == "ogbn-arxiv"
            #     or args.dataset == "Reddit"
            # ):
            #     part_edge_index = (
            #         part_edge_index.to_torch_sparse_coo_tensor().coalesce()
            #     )  # SparseTensorをCOO形式に変換し、圧縮
            #     part_edge_index = torch.stack(
            #         [
            #             part_edge_index.indices()[0],
            #             part_edge_index.indices()[1],
            #         ],
            #         dim=0,
            #     )  # rowとcolを結合して2行の形式に変換

            if args.flowgnn_debug:
                os.makedirs(
                    f"./pretrained_model/Output/l{layer_idx+1}",
                    exist_ok=True,
                )
                # Save logits to txt file
                logits_np = logits.detach().cpu().numpy()
                logits_flattened = logits_np.flatten()
                np.savetxt(
                    f"./pretrained_model/Output/l{layer_idx+1}/l{layer_idx+1}_part{part}_AXW.txt",
                    logits_flattened,
                    fmt="%.6f",
                )

            if args.inter_cluster is True:
                part_indices_tensor = part_indices.clone().detach().to(device)
                original_indices = (
                    torch.isin(new_part_indices, part_indices_tensor).nonzero().squeeze()
                )

                if len(original_indices) != len(part_indices_tensor):
                    print("Warning: new_part_indices contains elements not in part_indices_tensor.")
                    print("Removing these elements from new_part_indices.")
                    new_part_indices = new_part_indices[original_indices]
                    part_data_x = part_data_x[original_indices]
                    part_edge_index = filter_edges(part_edge_index, original_indices)
                    part_edge_index, _ = reindex_nodes(part_edge_index, original_indices)
                    part_val_mask = part_val_mask[original_indices]
                    part_test_mask = part_test_mask[original_indices]

                logits = logits[original_indices]

            if layer_idx < (args.num_layers - 1):
                expanded_output[part_indices] = logits
                del logits
                if device == "cuda:0":
                    torch.cuda.empty_cache()

            else:
                expanded_output = expanded_output[:, : args.num_classes]
                expanded_output[part_indices] = logits
                del logits
                if device == "cuda:0":
                    torch.cuda.empty_cache()

            # print("after expanded_outpu")
            #

        if args.flowgnn_debug:
            # node_part_tensor = torch.tensor(node_part_list)

            # # テキストファイルとして保存
            # with open(
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_incoming_outgoing_pid.txt",
            #     "w",
            # ) as f:
            #     for item in node_part_list:
            #         f.write(f"{item}\n")

            # torch.save(
            #     node_part_tensor,
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_incoming_outgoing_pid.bin",
            # )

            # incoming_qid_tensor = torch.tensor(incoming_qid_list)

            # with open(
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_incoming_outgoing_qid.txt",
            #     "w",
            # ) as f:
            #     for item in incoming_qid_tensor:
            #         f.write(f"{item}\n")

            # # バイナリファイルとして保存
            # torch.save(
            #     incoming_qid_tensor,
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_incoming_outgoing_qid.bin",
            # )

            # # outgoing_qid_listが8つのテンソルを含んでいるリストと仮定
            # outgoing_qid_list = [torch.tensor([i]) for i in range(outgoing_qid_list)]  # 例として0から7のテンソルを作成

            # テンソルを1列に結合
            # outgoing_qid_tensor = torch.cat(outgoing_qid_list)

            # # テキストファイルに保存
            # with open(
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_outgoing_qid.txt",
            #     "w",
            # ) as f:
            #     for item in outgoing_qid_tensor:
            #         f.write(f"{item.item()}\n")

            # # バイナリファイルとして保存
            # torch.save(
            #     outgoing_qid_tensor,
            #     f"./pretrained_model/Kmeans/l{layer_idx+1}/l{layer_idx+1}_outgoing_qid.bin",
            # )

            # if args.flowgnn_debug:
            # max_incoming_nodes = max(num_inter_nodes_list)
            # max_incoming_part = num_inter_nodes_list.index(max_incoming_nodes)

            # # 最大のoutgoing nodesを持つpartを記録
            # with open(
            #     "./pretrained_model/Kmeans/max_number_of_incoming_node.txt",
            #     "w",
            # ) as file:
            #     file.write(f"part {max_incoming_part}: {max_incoming_nodes}\n")

            with open(
                "./pretrained_model/Kmeans/num_of_incoming.txt",
                "w",
            ) as file:
                for part in range(n_parts):
                    file.write(f"{num_inter_nodes_list[part]}\n")

            with open("./pretrained_model/Kmeans/num_of_outgoing.txt", "w") as file:
                for part in range(n_parts):
                    file.write(f"{num_outgoing_nodes_list[part]}\n")

        intermediate_outputs.append(expanded_output)

        if args.inter_cluster is True or args.no_inter_cluster:
            avg_num_intra_nodes = sum(num_intra_nodes_counts) / len(num_intra_nodes_counts)
            avg_num_inter_nodes = sum(num_inter_nodes_counts) / len(num_inter_nodes_counts)
            avg_num_intra_and_inter_nodes = sum(num_intra_and_inter_nodes_counts) / len(
                num_intra_and_inter_nodes_counts
            )

    _, indices = torch.max(intermediate_outputs[-1], dim=1)

    if dataset == "ogbn-arxiv":
        correct_val = (indices[data.val_mask] == data.y.squeeze()[data.val_mask]).sum().item()
        correct_test = (indices[data.test_mask] == data.y.squeeze()[data.test_mask]).sum().item()
        total_val_correct += correct_val
        total_val_count += data.val_mask.sum().item()
        total_test_correct += correct_test
        total_test_count += data.test_mask.sum().item()
    else:
        if model_cfg == "GIN" or model_cfg == "dmGIN":
            correct_val = torch.sum(indices[data.val_mask] == data.y[data.val_mask])
            correct_test = torch.sum(indices[data.test_mask] == data.y[data.test_mask])
            total_val_correct += correct_val.item()
            total_val_count += data.val_mask.sum().item()
            total_test_correct += correct_test.item()
            total_test_count += data.test_mask.sum().item()
        else:
            correct_val = torch.sum(indices[data.val_mask] == data.y[data.val_mask])
            correct_test = torch.sum(indices[data.test_mask] == data.y[data.test_mask])
            total_val_correct += correct_val.item()
            total_val_count += data.val_mask.sum().item()
            total_test_correct += correct_test.item()
            total_test_count += data.test_mask.sum().item()

    acc_val = total_val_correct / total_val_count
    acc_test = total_test_correct / total_test_count

    print(acc_test)

    return (
        acc_val,
        acc_test,
        avg_num_intra_nodes,
        avg_num_inter_nodes,
        avg_num_intra_and_inter_nodes,
    )
