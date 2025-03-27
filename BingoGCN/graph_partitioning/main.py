import argparse

# import math
import os
import random
import re
import sys
import time
import warnings

import dgl
import numpy as np
import pandas as pd
import pymetis

# import pymetis
import torch

# from pymetis import part_graph
# from sklearn.cluster import KMeans
# from torch_cluster import radius_graph
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor
from tqdm import tqdm

from ..models.Dataloader import load_data, load_ogbn
from ..models.utils_gin import load_adj_raw_gin, load_data_gin
from .evaluation import evaluate_model_on_partitioned_graph
from .graph_utils import generate_adj_list, partition_graph
from .models import get_model
from .partition_utils import StreamingKMeans
from .utils import extract_log_data

seed = 0
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

sys.path.append("./")
start = time.time()
warnings.filterwarnings("ignore", category=FutureWarning)
# print(os.getcwd())
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description="Script Parameters")
parser.add_argument(
    "--n_parts",
    type=int,
    default=128,
    help="Number of parts to partition the graph into",
)
parser.add_argument("--M", type=int, default=16, help="Number of M")
parser.add_argument(
    "--repeat_times", type=int, default=10, help="Number of repeat times"
)
parser.add_argument("--inter_sparsity", type=float, default=0)
parser.add_argument(
    "--no_inter_cluster",
    action="store_true",
    default=False,
    help="Enable no inter cluster",
)
parser.add_argument("--init_mode", type=str, default="kaiming_uniform")

parser.add_argument(
    "--inter_cluster",
    action="store_true",
    default=False,
    help="Enable inter clutester",
)
parser.add_argument(
    "--random_sampling",
    action="store_true",
    default=False,
    help="Enable inter clutester with random_sampling",
)
parser.add_argument(
    "--topk_sampling",
    action="store_true",
    default=False,
    help="Enable inter clutester with topk_sampling",
)
parser.add_argument(
    "--topk_pruning",
    action="store_true",
    default=False,
    help="Enable inter clutester with topk_pruning",
)
parser.add_argument(
    "--no_edge_weight",
    action="store_true",
    default=False,
    help="Enable no_edge_weight",
)
parser.add_argument(
    "--pretrained_log",
    action="store",
    type=str,
    default=None,
    help="Path to tpretrained_log.",
)

parser.add_argument(
    "--dataset",
    action="store",
    type=str,
    default=None,
    help="dataset",
)

parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="GCN",
    help="model",
)

parser.add_argument(
    "--partial_kmeans",
    action="store_true",
    default=False,
    help="Enable partial_kmeans",
)
parser.add_argument(
    "--global_kmeans",
    action="store_true",
    default=False,
    help="Enable global_kmeans",
)
parser.add_argument(
    "--online_kmeans",
    action="store_true",
    default=False,
    help="Enable online_kmeans",
)
parser.add_argument(
    "--num_kmeans_clusters",
    action="store",
    type=int,
    default=10,
    help="num_kmeans_clusters",
)
parser.add_argument(
    "--kmeans_approximation",
    action="store",
    type=str,
    default="center",
    help="kmeans_approximation: center or average",
)
parser.add_argument(
    "--save_partition_bin",
    action="store_true",
    default=False,
    help="Enable save_partition_bin",
)
parser.add_argument(
    "--original_edge_weight",
    action="store_true",
    default=False,
    help="Enable using original_edge_weight",
)
parser.add_argument(
    "--download_prunedw",
    action="store_true",
    default=False,
    help="Enable using download_prunedw",
)

parser.add_argument(
    "--flowgnn_debug",
    action="store_true",
    default=False,
    help="flowgnn_debug",
)

parser.add_argument(
    "--online_metis",
    action="store_true",
    default=False,
    help="Enable online_metis",
)

parser.add_argument(
    "--streamingMETIS",
    action="store_true",
    default=False,
    help="Enable streaming METIS",
)

parser.add_argument(
    "--no_kmeans_approximation",
    action="store_true",
    default=False,
    help="Enable no_kmeans_approximation",
)

parser.add_argument(
    "--every_X_is_approximated",
    action="store_true",
    default=False,
    help="Enable every_X_is_approximated",
)

parser.add_argument(
    "--training_centroids",
    action="store_true",
    default=False,
    help="Enable training_centroids",
)

parser.add_argument(
    "--nmsparsity",
    action="store_true",
    default=False,
    help="Enable nmsparsity",
)

parser.add_argument(
    "--unstructured_for_last_layer",
    action="store_true",
    default=False,
    help="Enable unstructured_for_last_layer",
)

parser.add_argument(
    "--outgoing_kmeans",
    action="store_true",
    default=False,
    help="Enable outgoing_kmeans",
)

parser.add_argument(
    "--hierarchical_kmeans",
    action="store_true",
    default=False,
    help="Enable hierarchical_kmeans",
)

parser.add_argument(
    "--kmeans_before_relu",
    action="store_true",
    default=False,
    help="Enable kmeans_before_relu",
)

parser.add_argument(
    "--validate",
    action="store_true",
    default=False,
    help="Enable validate",
)

parser.add_argument(
    "--sparse_decay",
    action="store_true",
    default=False,
    help="Enable sparse_decay",
)


parser.add_argument("--kmeans_distance", type=str, default="l1")
parser.add_argument("--train_mode", type=str, default="normal")
parser.add_argument("--prunedw_path", type=str, default=None)
# parser.add_argument("--train_mode", type=str, default="score_only")

parser.add_argument("--nm_decay", type=float, default=0.00015)

parser.add_argument(
    "--fixed_centroid_ratio",
    action="store_true",
    default=False,
    help="Enable fixed_centroid_ratio",
)
parser.add_argument(
    "--enable_mask",
    action="store_true",
    default=False,
    help="Enable enable_mask",
)
parser.add_argument("--linear_sparsity", type=float, default=0.75)
parser.add_argument("--centroid_ratio", type=float, default=0.5)
parser.add_argument("--size_l1_cent_group", type=int, default=8)
parser.add_argument("--dim_hidden", type=int, default=192)

# Parse the arguments from the command line
args_cmd = parser.parse_args()

log_file = args_cmd.pretrained_log
log_filename = os.path.basename(log_file)
parts = log_filename.split("_")

with open(log_file, "r") as file:
    log_content = "".join(file.readlines()[:10])

sparsity_list_match = re.search(r"sparsity_list=\[(.*?)\]", log_content)
if sparsity_list_match:
    sparsity_list_str = sparsity_list_match.group(1)
    sparsity_list = [float(x.strip()) for x in sparsity_list_str.split(",")]
else:
    sparsity_list = None

linear_sparsity_match = re.search(r"linear_sparsity=(\d+\.?\d*)", log_content)
if linear_sparsity_match:
    linear_sparsity = float(linear_sparsity_match.group(1))
else:
    linear_sparsity = None
    # raise ValueError("sparsity not found in the log file.")

args = argparse.Namespace(
    num_feats=1433,  # 示例值，请根据您的需求进行更改
    dim_hidden=192,  # 示例值，请根据您的需求进行更改
    num_layers=2,  # 示例值，请根据您的需求进行更改
    transductive=True,
    init_mode_linear=None,
    init_mode="kaiming_uniform",
    init_mode_mask="kaiming_uniform",
    init_scale=1.0,
    init_scale_score=1.0,
    # sparsity_list = [0, 0, 0],
    num_classes=7,
    heads=1,
    enable_sw_mm=False,
    # BN_GIN = True,
    # type_norm = 'batch',
    type_norm="None",
    train_mode="normal",
    linear_sparsity=linear_sparsity,
    # train_mode = 'normal',
    bn_momentum=0.1,
    bn_track_running_stats=True,
    bn_affine=True,
    enable_abs_comp=True,
    random_seed=None,
    attack=None,
    attack_eps=None,
    evatime=False,
    evanum=50,
    sparsity_list=sparsity_list,
    enable_mask=True,
    num_mask=3,
    regular_weight_pruning=None,
    num_of_weight_blocks=1,
    global_th_for_rowbyrow=False,
    force_sparisty_distribution=False,
    adjsparsity_ratio=0,
    x_pruning_layer=0,
    enable_node_pruning=False,
    enable_feat_pruning=False,
    no_edge_weight=False,
    BN_GIN=False,
    validate=False,
    dropout=0,
    device=0,
)

args.init_mode = "kaiming_uniform"
args.dataset = args_cmd.dataset
dataset = args_cmd.dataset
model_types = [f"{args_cmd.model}"]
args.model = args_cmd.model
model_cfg = args_cmd.model
inter_cluster_sparsity = [0.0]

excel_data = []

with torch.no_grad():
    for model_cfg in model_types:
        if model_cfg == "GCN" or model_cfg == "dmGCN":
            args.modelname = "GCN"
        elif model_cfg == "dmGAT" or model_cfg == "GAT":
            args.modelname = "GAT"
        elif model_cfg == "dmGIN" or model_cfg == "GIN":
            args.modelname = "GIN"

        args.weight_decay = 0.0
        args.spadj = False
        args.adjsparsity_ratio = 0.0
        args.gra_part = True
        args.inter_cluster = args_cmd.inter_cluster
        args.n_parts = args_cmd.n_parts
        args.inter_sparsity = args_cmd.inter_sparsity
        args.no_inter_cluster = args_cmd.no_inter_cluster
        args.random_sampling = args_cmd.random_sampling
        args.topk_sampling = args_cmd.topk_sampling
        args.topk_pruning = args_cmd.topk_pruning
        args.linear_sparsity = args_cmd.linear_sparsity
        args.pretrained_log = args_cmd.pretrained_log
        args.num_kmeans_clusters = args_cmd.num_kmeans_clusters
        args.kmeans_approximation = args_cmd.kmeans_approximation
        args.save_partition_bin = args_cmd.save_partition_bin
        args.original_edge_weight = args_cmd.original_edge_weight
        args.download_prunedw = args_cmd.download_prunedw
        args.global_kmeans = args_cmd.global_kmeans
        args.online_kmeans = args_cmd.online_kmeans
        args.partial_kmeans = args_cmd.partial_kmeans
        args.flowgnn_debug = args_cmd.flowgnn_debug
        args.outgoing_kmeans = args_cmd.outgoing_kmeans
        args.nmsparsity = args_cmd.nmsparsity
        args.kmeans_before_relu = args_cmd.kmeans_before_relu
        args.unstructured_for_last_layer = args_cmd.unstructured_for_last_layer
        args.M = args_cmd.M
        args.hierarchical_kmeans = args_cmd.hierarchical_kmeans
        args.online_metis = args_cmd.online_metis
        args.no_kmeans_approximation = args_cmd.no_kmeans_approximation
        args.nm_decay = args_cmd.nm_decay
        args.init_mode = args_cmd.init_mode
        args.sparse_decay = args_cmd.sparse_decay
        args.train_mode = args_cmd.train_mode
        args.streamingMETIS = args_cmd.streamingMETIS
        args.repeat_times = args_cmd.repeat_times
        args.training_centroids = args_cmd.training_centroids
        args.every_X_is_approximated = args_cmd.every_X_is_approximated
        args.validate = args_cmd.validate
        args.intra_cluster_sparse = False
        args.no_edge_weight = False
        args.kmeans_distance = args_cmd.kmeans_distance
        args.model = args_cmd.model
        args.fixed_centroid_ratio = args_cmd.fixed_centroid_ratio
        args.centroid_ratio = args_cmd.centroid_ratio
        args.size_l1_cent_group = args_cmd.size_l1_cent_group
        args.prunedw_path = args_cmd.prunedw_path
        args.train_mode = args_cmd.train_mode
        if args.train_mode == "normal":
            args.enable_mask = False
            args.linear_sparsity = 0

        if args.flowgnn_debug:
            kmeans_dir = "./pretrained_model/Kmeans"
            models_dir = "./pretrained_model/Models"
            models_dir = "./pretrained_model/Input"

            os.makedirs(kmeans_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)

        if dataset == "Cora":
            args.num_feats = 1433
            args.dim_hidden = 192
            args.num_layers = 3
            args.num_classes = 7
            args.dropout = 0.6
        elif dataset == "Citeseer":
            args.num_feats = 3703
            args.dim_hidden = 192
            args.num_layers = 3
            args.num_classes = 6
            args.dropout = 0.7
        elif dataset == "Pubmed":
            args.num_feats = 500
            args.dim_hidden = 192
            args.num_layers = 3
            args.num_classes = 3
            args.dropout = 0.5
        elif dataset == "ogbn-arxiv":
            args.num_feats = 128
            args.dim_hidden = 192
            args.num_nodes = 169343
            args.num_layers = 4
            args.num_classes = 40
            args.dropout = 0
            args.type_norm = "batch"
        elif dataset == "Reddit":
            args.num_feats = 602
            args.dim_hidden = 192
            args.num_layers = 4
            args.num_classes = 41
            args.dropout = 0
            args.type_norm = "batch"
            args.num_nodes = 232965

        acc_mean, acc_std, best_acc, best_model_path = extract_log_data(
            log_file
        )

        if best_model_path is None:
            continue
        if best_model_path is not None:
            model_paths = [
                path.replace("best", f"{i}.best")
                for i in range(0, 3)
                for path in [best_model_path]
            ]
            val_accuracies = []
            test_accuracies = []

            for path in model_paths:
                if os.path.exists(path):
                    if dataset == "ogbn-arxiv":
                        data, split_idx = load_ogbn(
                            dataset, None, None, sparse_tensor=True
                        )
                        data = data.cuda()
                        num_nodes = data.num_nodes
                        data.train_mask = torch.zeros(
                            num_nodes, dtype=torch.bool
                        )
                        data.test_mask = torch.zeros(
                            num_nodes, dtype=torch.bool
                        )
                        data.val_mask = torch.zeros(
                            num_nodes, dtype=torch.bool
                        )
                        data.train_mask[split_idx["train"]] = True
                        data.test_mask[split_idx["test"]] = True
                        data.val_mask[split_idx["valid"]] = True
                    else:
                        data = load_data(
                            dataset,
                            args.random_seed,
                            args.attack,
                            args.attack_eps,
                        ).cuda()

                    for args.intra_sparsity_ratio in inter_cluster_sparsity:
                        if model_cfg == "dmGIN" or model_cfg == "GIN":

                            g = dgl.DGLGraph()
                            (
                                ginadj,
                                ginfeatures,
                                ginlabels,
                                ginidx_train,
                                ginidx_val,
                                ginidx_test,
                            ) = load_data_gin(dataset.lower())
                            ginadj = load_adj_raw_gin(dataset.lower())
                            ginadj = ginadj.tocoo()
                            ginnode_num = ginfeatures.size()[0]
                            ginclass_num = ginlabels.numpy().max() + 1
                            g.add_nodes(ginnode_num)
                            g.add_edges(ginadj.row, ginadj.col)
                            g = g.to("cuda")
                            g.adjacency_matrix = (
                                torch.tensor(ginadj.toarray())
                                .to("cuda")
                                .float()
                            )
                            pretrained_model = get_model(
                                args, model_cfg, g
                            ).cuda()
                            edge_index = torch.stack(g.edges(), dim=0).to(
                                "cuda"
                            )
                            data.edge_index = edge_index
                            # Create boolean masks for val and test nodes
                            val_mask = torch.zeros(
                                ginnode_num, dtype=torch.bool
                            )
                            val_mask[ginidx_val] = True
                            test_mask = torch.zeros(
                                ginnode_num, dtype=torch.bool
                            )
                            test_mask[ginidx_test] = True

                            ginidx_val = val_mask.to("cuda")
                            ginidx_test = test_mask.to("cuda")
                        else:
                            pretrained_model = get_model(
                                args, model_cfg
                            ).cuda()

                        pretrained_weights = torch.load(path)
                        model_state_dict = pretrained_weights[
                            "model_state_dict"
                        ]
                        pretrained_model.load_state_dict(model_state_dict)

                        if dataset == "ogbn-arxiv":
                            pass
                            # edge_index = data.edge_index
                            # indices = edge_index
                            # values = torch.ones(
                            #     indices.shape[1], dtype=torch.bool
                            # ).cuda()
                            # size = torch.Size([num_nodes, num_nodes])
                            # adj_matrix = torch.sparse_coo_tensor(
                            #     indices, values, size, dtype=torch.bool
                            # ).cuda()
                        elif args.dataset == "Reddit":
                            pass
                            # if isinstance(data.edge_index, SparseTensor):
                            #     adj_matrix = data.edge_index
                            # else:
                            #     num_nodes = data.num_nodes
                            #     edge_index = data.edge_index
                            #     indices = edge_index
                            #     values = torch.ones(
                            #         indices.shape[1], dtype=torch.bool
                            #     ).cuda()
                            #     size = torch.Size([num_nodes, num_nodes])
                            #     adj_matrix = torch.sparse_coo_tensor(
                            #         indices, values, size, dtype=torch.bool
                            #     ).cuda()

                            # edges = (
                            #     adj_matrix.coalesce().indices()
                            #     if not isinstance(
                            #         data.edge_index, SparseTensor
                            #     )
                            #     else torch.stack(adj_matrix.coo()[:2], dim=0)
                            # )
                            # adj_list = generate_adj_list_ogbn(adj_matrix)
                        else:
                            if model_cfg == "GIN":
                                adj_matrix = g.adjacency_matrix

                            else:
                                num_nodes = data.num_nodes
                                edge_index = data.edge_index
                                indices = edge_index
                                values = torch.ones(
                                    indices.shape[1], dtype=torch.bool
                                ).cuda()
                                size = torch.Size([num_nodes, num_nodes])
                                adj_matrix = torch.zeros(
                                    (data.num_nodes, data.num_nodes),
                                    dtype=torch.float32,
                                ).cuda()
                                adj_matrix[
                                    data.edge_index[0], data.edge_index[1]
                                ] = 1

                        if args.spadj is True:
                            adjsparsity_ratio = args.adjsparsity_ratio
                            print("adjsparsity_ratio is", adjsparsity_ratio)
                            edges = adj_matrix.nonzero(as_tuple=True)
                            num_edges_to_remove = int(
                                adjsparsity_ratio * len(edges[0])
                            )
                            edges_to_remove = torch.randperm(
                                len(edges[0]), device="cpu"
                            )[:num_edges_to_remove]
                            for idx in edges_to_remove:
                                adj_matrix[edges[0][idx], edges[1][idx]] = 0

                        if args.gra_part is True:

                            if args.streamingMETIS:
                                print(args)

                                def partition_train_graph(
                                    data,
                                    edge_index,
                                    train_mask,
                                    val_mask,
                                    test_mask,
                                    args,
                                ):
                                    # device = "cpu"

                                    # Define paths for saving and loading results
                                    sorted_indices_path = f"streaming_sorted_indices_p{args.n_parts}.txt"
                                    sorted_membership_path = f"streaming_sorted_membership_p{args.n_parts}.txt"
                                    centroids_all_path = f"streaming_centroids_all_{args.num_kmeans_clusters}.npy"

                                    if os.path.exists(
                                        sorted_indices_path
                                    ) and os.path.exists(
                                        sorted_membership_path
                                    ):
                                        sorted_indices = torch.tensor(
                                            np.loadtxt(
                                                sorted_indices_path, dtype=int
                                            ),
                                            device="cpu",
                                        )
                                        sorted_membership = torch.tensor(
                                            np.loadtxt(
                                                sorted_membership_path,
                                                dtype=int,
                                            ),
                                            device="cpu",
                                        )
                                        if (
                                            args.training_centroids
                                            and os.path.exists(
                                                centroids_all_path
                                            )
                                        ):
                                            centroids_all = np.load(
                                                centroids_all_path,
                                                allow_pickle=True,
                                            )
                                            centroids_all = [
                                                torch.tensor(
                                                    centroid, device="cpu"
                                                )
                                                for centroid in centroids_all
                                            ]
                                        else:
                                            centroids_all = None
                                        return (
                                            sorted_indices,
                                            sorted_membership,
                                            centroids_all,
                                        )

                                    train_edge_index, _ = subgraph(
                                        train_mask,
                                        edge_index.to("cpu"),
                                        relabel_nodes=True,
                                    )
                                    membership = np.loadtxt(
                                        "train_membership_512.txt", dtype=int
                                    )

                                    # Output the number of intra nodes and inter nodes for each membership
                                    for part in range(args.n_parts):
                                        part_mask = (
                                            torch.tensor(membership) == part
                                        )
                                        part_mask = part_mask.to(
                                            "cpu"
                                        )  # Move part_mask to GPU
                                        part_nodes = torch.where(part_mask)[0]
                                        part_edge_index, _ = subgraph(
                                            part_mask,
                                            train_edge_index,
                                            relabel_nodes=True,
                                        )

                                        intra_nodes = part_nodes.shape[0]

                                        other_parts_mask = (
                                            torch.tensor(membership) != part
                                        )
                                        other_parts_mask = other_parts_mask.to(
                                            "cpu"
                                        )
                                        # Generate mask considering bidirectional edges
                                        inter_nodes_mask = (
                                            (
                                                train_edge_index[0].unsqueeze(
                                                    1
                                                )
                                                == part_nodes
                                            ).any(1)
                                            & other_parts_mask[
                                                train_edge_index[1]
                                            ]
                                        ) | (
                                            (
                                                train_edge_index[1].unsqueeze(
                                                    1
                                                )
                                                == part_nodes
                                            ).any(1)
                                            & other_parts_mask[
                                                train_edge_index[0]
                                            ]
                                        )  # noqa
                                        inter_nodes_indices = torch.cat(
                                            [
                                                train_edge_index[0][
                                                    inter_nodes_mask
                                                ],
                                                train_edge_index[1][
                                                    inter_nodes_mask
                                                ],
                                            ]
                                        )
                                        inter_nodes = (
                                            inter_nodes_indices.unique().shape[
                                                0
                                            ]
                                        )

                                        print(
                                            f"Train Group {part}: Intra nodes = {intra_nodes}, Inter nodes = {inter_nodes}"
                                        )

                                    membership_tensor = torch.tensor(
                                        membership, device="cpu"
                                    )
                                    part_node_counts = torch.bincount(
                                        membership_tensor,
                                        minlength=args.n_parts,
                                    )

                                    if args.training_centroids:
                                        centroids_all = [None] * args.n_parts

                                        for i in range(args.n_parts):
                                            centroids = np.loadtxt(
                                                f"layer0_centroids/partition{i}_centroids.txt"
                                            )
                                            centroids_all[i] = (
                                                torch.from_numpy(centroids).to(
                                                    "cpu"
                                                )
                                            )

                                        streaming_kmeans = StreamingKMeans(
                                            args.num_kmeans_clusters,
                                            args.n_parts,
                                            "cpu",
                                        )

                                    remaining_mask = torch.zeros(
                                        data.num_nodes,
                                        dtype=torch.bool,
                                        device="cpu",
                                    )
                                    remaining_mask[val_mask] = True
                                    remaining_mask[test_mask] = True
                                    remaining_nodes = torch.nonzero(
                                        remaining_mask, as_tuple=True
                                    )[0]

                                    subgraph_assignments = -torch.ones(
                                        data.num_nodes,
                                        dtype=torch.long,
                                        device="cpu",
                                    )
                                    subgraph_assignments[train_mask] = (
                                        membership_tensor
                                    )

                                    unassigned_nodes = remaining_nodes
                                    total_unassigned = unassigned_nodes.numel()

                                    with tqdm(
                                        total=total_unassigned,
                                        desc="Assigning nodes",
                                    ) as pbar:
                                        while unassigned_nodes.numel() > 0:
                                            remaining_unassigned_nodes = []
                                            for node_id in unassigned_nodes:
                                                neighbors = torch.unique(
                                                    edge_index[
                                                        :,
                                                        (
                                                            edge_index[0]
                                                            == node_id
                                                        )
                                                        | (
                                                            edge_index[1]
                                                            == node_id
                                                        ),
                                                    ].flatten()
                                                )

                                                neighbor_subgraphs = (
                                                    subgraph_assignments[
                                                        neighbors
                                                    ]
                                                )
                                                valid_subgraphs = (
                                                    neighbor_subgraphs[
                                                        neighbor_subgraphs
                                                        != -1
                                                    ]
                                                )

                                                if valid_subgraphs.numel() > 0:
                                                    subgraph_counts = torch.bincount(
                                                        valid_subgraphs,
                                                        minlength=args.n_parts,
                                                    )
                                                    max_count = (
                                                        subgraph_counts.max().item()
                                                    )
                                                    max_count_subgraphs = (
                                                        torch.where(
                                                            subgraph_counts
                                                            == max_count
                                                        )[0]
                                                    )

                                                    if (
                                                        max_count_subgraphs.numel()
                                                        > 1
                                                    ):
                                                        min_node_count_subgraph = max_count_subgraphs[
                                                            part_node_counts[
                                                                max_count_subgraphs
                                                            ].argmin()
                                                        ]
                                                        assigned_subgraph = min_node_count_subgraph
                                                    else:
                                                        assigned_subgraph = max_count_subgraphs[
                                                            0
                                                        ]

                                                    subgraph_assignments[
                                                        node_id
                                                    ] = assigned_subgraph
                                                    part_node_counts[
                                                        assigned_subgraph
                                                    ] += 1
                                                else:
                                                    remaining_unassigned_nodes.append(
                                                        node_id
                                                    )

                                                for part in torch.unique(
                                                    valid_subgraphs
                                                ):
                                                    if (
                                                        part
                                                        != assigned_subgraph
                                                    ):
                                                        node_features = data.x[
                                                            node_id
                                                        ].unsqueeze(0)
                                                        if (
                                                            args.training_centroids
                                                        ):
                                                            (
                                                                updated_centroids,
                                                                cluster_counts,
                                                            ) = streaming_kmeans.fit(
                                                                centroids_all[
                                                                    part
                                                                ],
                                                                node_features,
                                                                part,
                                                            )
                                                            centroids_all[
                                                                part
                                                            ] = updated_centroids

                                                pbar.update(1)

                                            unassigned_nodes = torch.tensor(
                                                remaining_unassigned_nodes,
                                                device="cpu",
                                            )

                                    # Save the final results to text files

                                    sorted_indices = torch.argsort(
                                        subgraph_assignments
                                    )
                                    sorted_membership = subgraph_assignments[
                                        sorted_indices
                                    ]

                                    if not args.training_centroids:
                                        centroids_all = None

                                    np.savetxt(
                                        sorted_indices_path,
                                        sorted_indices.cpu().numpy(),
                                        fmt="%d",
                                    )
                                    np.savetxt(
                                        sorted_membership_path,
                                        sorted_membership.cpu().numpy(),
                                        fmt="%d",
                                    )
                                    if args.training_centroids:
                                        np.save(
                                            centroids_all_path,
                                            [
                                                centroid.cpu().numpy()
                                                for centroid in centroids_all
                                            ],
                                        )

                                    return (
                                        sorted_indices,
                                        sorted_membership,
                                        centroids_all,
                                    )

                                device = "cpu"

                                train_mask = split_idx["train"].to(device)
                                val_mask = split_idx["valid"].to(device)
                                test_mask = split_idx["test"].to(device)

                                # データをGPUに移動
                                data = data.to(device)

                                # グラフを分割
                                (
                                    sorted_indices,
                                    sorted_membership,
                                    centroids_all,
                                ) = partition_train_graph(
                                    data,
                                    data.edge_index,
                                    train_mask,
                                    val_mask,
                                    test_mask,
                                    args,
                                )

                                # データをCPUに移動
                                data = data.to("cpu")
                                pretrained_model.eval()
                                # 分割されたグラフでモデルを評価
                                if model_cfg == "GIN":
                                    (
                                        acc_val,
                                        acc_test,
                                        avg_num_intra_nodes,
                                        avg_num_inter_nodes,
                                        avg_num_intra_and_inter_nodes,
                                    ) = evaluate_model_on_partitioned_graph(
                                        pretrained_model,
                                        data,
                                        sorted_indices,
                                        sorted_membership,
                                        args.n_parts,
                                        aux_adj=adj_matrix,
                                        ori_membership=sorted_membership.cpu().numpy(),
                                        args=args,
                                        g=g,
                                        ginfeatures=ginfeatures,
                                        ginlabels=ginlabels,
                                        ginidx_train=ginidx_train,
                                        ginidx_val=ginidx_val,
                                        ginidx_test=ginidx_test,
                                        dataset=dataset,
                                        model_cfg=model_cfg,
                                    )
                                else:
                                    (
                                        acc_val,
                                        acc_test,
                                        avg_num_intra_nodes,
                                        avg_num_inter_nodes,
                                        avg_num_intra_and_inter_nodes,
                                    ) = evaluate_model_on_partitioned_graph(
                                        pretrained_model,
                                        data,
                                        sorted_indices,
                                        sorted_membership,
                                        args.n_parts,
                                        aux_adj=adj_matrix,
                                        ori_membership=sorted_membership.cpu().numpy(),
                                        args=args,
                                        dataset=dataset,
                                        model_cfg=model_cfg,
                                        centroids=centroids_all,
                                    )

                                val_accuracies.append(acc_val)
                                test_accuracies.append(acc_test)

                            else:
                                # Graph Partitioning
                                if isinstance(data.edge_index, SparseTensor):
                                    csr_matrix = data.edge_index.to_scipy(
                                        layout="csr"
                                    )
                                    n_vertices = csr_matrix.shape[0]
                                    xadj = csr_matrix.indptr.tolist()
                                    adjncy = csr_matrix.indices.tolist()
                                    cuts, membership = pymetis.part_graph(
                                        args.n_parts, xadj=xadj, adjncy=adjncy
                                    )
                                    membership = np.array(membership)
                                    edges = data.edge_index.coo()[:2]
                                else:
                                    edges = adj_matrix.nonzero(as_tuple=True)
                                    adj_list = generate_adj_list(adj_matrix)
                                    n_cuts, membership = partition_graph(
                                        adj_list, args.n_parts
                                    )
                                    membership = np.array(membership)
                                sorted_indices = torch.argsort(
                                    torch.from_numpy(membership).to(
                                        edges[0].device
                                    )
                                )
                                # sorted_edges = torch.stack(
                                #     [sorted_indices[edges[0]], sorted_indices[edges[1]]]
                                # )
                                sorted_membership = torch.from_numpy(
                                    membership
                                ).to(edges[0].device)[sorted_indices]

                                pretrained_model.eval()

                                # 通常のMETISの場合の評価
                                if model_cfg == "GIN":
                                    (
                                        acc_val,
                                        acc_test,
                                        avg_num_intra_nodes,
                                        avg_num_inter_nodes,
                                        avg_num_intra_and_inter_nodes,
                                    ) = evaluate_model_on_partitioned_graph(
                                        pretrained_model,
                                        data,
                                        sorted_indices,
                                        sorted_membership,
                                        args.n_parts,
                                        aux_adj=adj_matrix,
                                        ori_membership=membership,
                                        args=args,
                                        g=g,
                                        ginfeatures=ginfeatures,
                                        ginlabels=ginlabels,
                                        ginidx_train=ginidx_train,
                                        ginidx_val=ginidx_val,
                                        ginidx_test=ginidx_test,
                                        dataset=dataset,
                                        model_cfg=model_cfg,
                                    )
                                else:
                                    (
                                        acc_val,
                                        acc_test,
                                        avg_num_intra_nodes,
                                        avg_num_inter_nodes,
                                        avg_num_intra_and_inter_nodes,
                                    ) = evaluate_model_on_partitioned_graph(
                                        pretrained_model,
                                        data,
                                        sorted_indices,
                                        sorted_membership,
                                        args.n_parts,
                                        aux_adj=None,
                                        ori_membership=membership,
                                        args=args,
                                        dataset=dataset,
                                        model_cfg=model_cfg,
                                    )
                                val_accuracies.append(acc_val)
                                test_accuracies.append(acc_test)
                                # print(acc_test)

                        else:
                            raise NotImplementedError

            # 精度の平均を計算
            avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
            avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
            formatted_acc_val = "{:.5f}".format(avg_val_accuracy)
            formatted_acc_test = "{:.5f}".format(avg_test_accuracy)

            end = time.time()
            time_diff = end - start

            if (
                args.partial_kmeans
                or args.global_kmeans
                and (avg_num_inter_nodes > args.num_kmeans_clusters)
            ):
                avg_num_inter_nodes = args.num_kmeans_clusters
                avg_num_intra_and_inter_nodes = (
                    avg_num_intra_nodes + avg_num_inter_nodes
                )

            record = {
                "Dataset": dataset,
                "Model Name": model_cfg,
                "No partition Test Acc": acc_mean,
                "Test Acc": formatted_acc_test,
                "Avg num intra nodes": avg_num_intra_nodes,
                "Avg num inter nodes": avg_num_inter_nodes,
                "Avg num intra and inter nodes": avg_num_intra_and_inter_nodes,
                "Execution time": time_diff,
            }
            excel_data.append(record)

    print(test_accuracies)
    df = pd.DataFrame(excel_data)
    print(record)
