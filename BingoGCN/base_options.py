# import argparse


class BaseOptions:

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self, parser):

        parser.add_argument(
            "--dataset",
            type=str,
            default="Cora",
            required=False,
            help="The input dataset.",
            # choices=[
            #     "Cora",
            #     "Citeseer",
            #     "Pubmed",
            #     "ogbn-arxiv",
            #     # "CoauthorCS",
            #     # "CoauthorPhysics",
            #     # "AmazonComputers",
            #     # "AmazonPhoto",
            #     # "TEXAS",
            #     # "WISCONSIN",
            #     # "ACTOR",
            #     # "CORNELL",
            #     "Reddit",
            #     "ogbg-molhiv",
            #     "ogbg-molbace",
            # ],
        )
        # build up the common parameter
        parser.add_argument("--random_seed", type=int, default=None)
        parser.add_argument("--resume", action="store_true", default=False)
        parser.add_argument(
            "--cuda",
            type=bool,
            default=True,
            required=False,
            help="run in cuda mode",
        )

        parser.add_argument(
            "--type_model",
            type=str,
            default="GCN",
            choices=[
                "GCN",
                "GAT",
                "SGC",
                "GIN",
                "GCNII",
                "DAGNN",
                "GPRGNN",
                "APPNP",
                "JKNet",
                "DeeperGCN",
                "dmGCN",
                "dmGAT",
                "dmGIN",
                "FAdmGCN",
                "GCN_graph_UGTs",
            ],
        )
        parser.add_argument(
            "--dropout", type=float, default=0, help="dropout for GCN"
        )
        # parser.add_argument('--embedding_dropout', type=float, default=0,
        #                    help='dropout for embeddings')
        parser.add_argument(
            "--lr", type=float, default=0.01, help="learning rate"
        )
        parser.add_argument(
            "--weight_decay", type=float, default=5e-4, help="weight decay"
        )  # 5e-4

        parser.add_argument(
            "--transductive",
            type=bool,
            default=True,
            help="transductive or inductive setting",
        )
        parser.add_argument(
            "--activation", type=str, default="relu", required=False
        )

        parser.add_argument("--type_norm", type=str, default="None")
        parser.add_argument("--command", type=str)
        # parser.add_argument('--config', type=str, help='file path for YAML configure file')

        # parser.add_argument('--accum_grad', type=int, default=1)
        parser.add_argument("--force_restart", action="store_true")
        parser.add_argument("--seed_by_time", type=bool, default=True)
        parser.add_argument("--print_train_loss", type=bool, default=False)
        parser.add_argument("--num_gpus", type=int, default=1)

        # for config.yaml
        # parser.add_argument('--save_best_model', type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="__outputs__")
        parser.add_argument("--sync_dir", type=str, default="__sync__")
        parser.add_argument("--save_best_model", type=bool, default=True)
        parser.add_argument("--optimizer", type=str, default="Adam")
        # parser.add_argument("--scheduler", type=str, default="CustomCosineLR")
        parser.add_argument("--scheduler", type=str, default="CustomCosineLR")
        parser.add_argument("--warm_epochs", type=int, default=0)
        parser.add_argument("--finetuning_epochs", type=int, default=0)
        parser.add_argument("--finetuning_lr", type=float, default=None)
        # parser.add_argument('--step',type=float,default=1)
        parser.add_argument(
            "--lr_milestones", type=list, default=[900, 1000, 1100]
        )
        parser.add_argument("--checkpoint_epochs", type=list, default=[])
        parser.add_argument("--multisteplr_gamma", type=float, default=0.1)
        parser.add_argument(
            "--learning_framework", type=str, default="SupervisedLearning"
        )
        parser.add_argument(
            "--bn_track_running_stats", type=bool, default=True
        )
        parser.add_argument("--bn_affine", type=bool, default=True)

        parser.add_argument("--bn_momentum", type=float, default=0.1)

        parser.add_argument("--init_mode", type=str, default="kaiming_uniform")
        parser.add_argument(
            "--init_mode_mask", type=str, default="kaiming_uniform"
        )
        parser.add_argument("--init_mode_linear", type=str, default=None)
        parser.add_argument("--init_scale", type=float, default=1.0)
        parser.add_argument("--init_scale_score", type=float, default=1.0)
        parser.add_argument("--heads", type=int, default=1)

        # paraser.add multi mask
        # parser.add_argument('--threshold_list',type=list, default=None)
        parser.add_argument(
            "--sparsity_list",
            nargs="+",
            type=float,
            default=None,
            help="List of threshold values.",
        )

        parser.add_argument("--num_mask", type=int, default=3)
        parser.add_argument(
            "--enable_mask",
            action="store_true",
            default=False,
            help="Enable multi-mask.",
        )
        parser.add_argument(
            "--enable_abs_comp",
            action="store_true",
            default=True,
            help="Enable abs in percentile.",
        )
        parser.add_argument(
            "--validate",
            action="store_true",
            default=False,
            help="Enable validate .",
        )
        parser.add_argument(
            "--pretrained_path",
            action="store",
            type=str,
            default=None,
            help="Path to the pretrained model.",
        )

        parser.add_argument(
            "--prunedw_path",
            action="store",
            type=str,
            default=None,
            help="Path to the prunedw_path model.",
        )

        # add
        parser.add_argument(
            "--enable_sw_mm",
            action="store_true",
            default=False,
            help="Enable single weight but multi scores for folded block.",
        )
        parser.add_argument(
            "--BN_GIN",
            action="store_true",
            default=False,
            help="Enable BN in GIN.",
        )

        # test evaluate
        parser.add_argument(
            "--evatime", action="store_true", default=False, help="evatime."
        )
        parser.add_argument(
            "--drawadj", action="store_true", default=False, help="drawpics."
        )
        parser.add_argument("--evanum", type=int, default=50)

        # METIS
        parser.add_argument(
            "--METIS",
            action="store_true",
            default=False,
            help="Enable METIS Graph Partitioning in GNNs.",
        )
        # Sparseadj
        parser.add_argument(
            "--spadj",
            action="store_true",
            default=False,
            help="Enable sparse adj  in GNNs.",
        )
        parser.add_argument("--adjsparsity_ratio", type=float, default=0.0)

        parser.add_argument(
            "--enable_feat_pruning",
            action="store_true",
            default=False,
            help="Enable features dim pruning.(columnwise pruning)",
        )
        parser.add_argument("--featsparsity_ratio", type=float, default=0.0)

        parser.add_argument(
            "--enable_node_pruning",
            action="store_true",
            default=False,
            help="Enable node dim pruning.(rowwise pruning)",
        )

        parser.add_argument("--x_pruning_layer", type=int, default=0)

        # parser.add_argument(
        #     "--regular_weight_pruning",
        #     action="store_true",
        #     default=False,
        #     help="Enable weight regular pruning for SLT (column-wise pruning)",
        # )

        parser.add_argument(
            "--regular_weight_pruning",
            type=str,
            default=None,
            required=False,
            help="regular_weight_pruning.",
            choices=[
                "block",
                "width",
            ],
        )

        parser.add_argument(
            "--num_of_weight_blocks",
            type=int,
            default=1,
            help="number of weight blocks for regular weight pruning (column-wise pruning)",
        )

        parser.add_argument(
            "--global_th_for_rowbyrow",
            action="store_true",
            default=False,
            help="Enable global_for_rowbyrow.(rowwise weight regular pruning)",
        )

        parser.add_argument(
            "--download_prunedw",
            action="store_true",
            default=False,
            help="Enable download_prunedw",
        )

        # parser.add_argument(
        #     "--global_th_for_rowbyrow",
        #     action="store_true",
        #     default=False,
        #     help="Enable global_for_rowbyrow.(rowwise weight regular pruning)",
        # )

        parser.add_argument(
            "--gra_part",
            action="store_true",
            default=False,
            help="gra_part",
        )

        parser.add_argument(
            "--no_edge_weight",
            action="store_true",
            default=False,
            help="no_edge_weight",
        )

        parser.add_argument(
            "--sparse_tensor",
            action="store_true",
            default=False,
            help="sparse_tenso",
        )

        parser.add_argument(
            "--original_edge_weight",
            action="store_true",
            default=False,
            help="original_edge_weight",
        )

        parser.add_argument(
            "--only_train_data",
            action="store_true",
            default=False,
            help="only_train_data for training",
        )

        parser.add_argument(
            "--flowgnn_debug",
            action="store_true",
            default=False,
            help="flowgnn_debug",
        )

        parser.add_argument(
            "--nmsparsity",
            action="store_true",
            default=False,
            help="nmsparsity",
        )

        parser.add_argument(
            "--M",
            type=int,
            default=4,
            help="number of M of N:M sparsity",
        )

        parser.add_argument(
            "--unstructured_for_last_layer",
            action="store_true",
            default=False,
            help="unstructured_for_last_layer",
        )

        parser.add_argument(
            "--xor_seed_using_instance_number",
            action="store_true",
            default=False,
            help="xor_seed_using_instance_number",
        )

        parser.add_argument(
            "--outgoing_centroids",
            action="store_true",
            default=False,
            help="outgoing_centroids",
        )

        parser.add_argument("--folded_layer", type=int, default=0)

        # BN_track_running_stats
        parser.add_argument(
            "--BN_track_running_stats",
            # action="store_true",
            default=True,
            help="BN_track_running_stats",
        )

        # folded_smの例
        parser.add_argument(
            "--folded_SM",
            action="store_false",
            default=False,
            help="Enbale folded",
        )
        parser.add_argument(
            "--half_folded",
            action="store_true",
            default=False,
            help="Enbale half-folded",
        )
        parser.add_argument("--batch_size", type=int, default=1024)

        parser.add_argument(
            "--elastic_net",
            action="store_true",
            default=False,
            help="Enbale Elastic Net",
        )
        parser.add_argument("--elastic_lambda", type=float, default=5e-8)
        parser.add_argument("--l1_ratio", type=float, default=0.5)
        parser.add_argument(
            "--no_norm", action="store_true", default=False, help="no_norm"
        )
        parser.add_argument(
            "--adp", action="store_true", default=False, help="adp_subnets"
        )
        # parser.add_argument('--ugts', action='store_true', default=False, help='emb to linear')
        parser.add_argument(
            "--SLT_Bonder",
            action="store_true",
            default=False,
            help="encoder to SLT",
        )
        parser.add_argument(
            "--SLTAtom",
            action="store_true",
            default=False,
            help="is_weight atom emb",
        )
        parser.add_argument(
            "--SLTBond",
            action="store_true",
            default=False,
            help="is_weight bond emb",
        )
        parser.add_argument(
            "--SLTRoot",
            action="store_true",
            default=False,
            help="is_weight bond emb",
        )
        parser.add_argument(
            "--SLTAtom_ini",
            type=str,
            default="default",
            help="change weight values",
        )
        parser.add_argument(
            "--SLTBond_ini",
            type=str,
            default="default",
            help="change weight values",
        )
        parser.add_argument(
            "--SLTRoot_ini",
            type=str,
            default="default",
            help="change weight values",
        )
        parser.add_argument(
            "--dense_for_last_layer",
            action="store_true",
            default=False,
            help="Enable dense_for_last_layer",
        )

        # parser.add_argument('--attack',action='store_true')
        parser.add_argument(
            "--local_pruning",
            action="store_true",
            default=False,
            help="Enable local_pruning",
        )

        parser.add_argument(
            "--local_sparsity_list",
            type=str,
            default=None,
            help="List of local_sparsity.",
        )

        parser.add_argument(
            "--inference_time",
            action="store_true",
            default=False,
            help="calc inference time for every epochs",
        )

        parser.add_argument(
            "--sparsity_profiling",
            action="store_true",
            default=False,
            help="sparsity_profiling",
        )

        parser.add_argument("--nm_decay", type=float, default=0.0015)

        args = parser.parse_args()
        args = self.reset_dataset_dependent_parameters(args)
        args = self.reset_train_mode_parameters(args)

        return args

    # setting the common hyperparameters used for comparing different methods of a trick
    def reset_train_mode_parameters(self, args):

        if args.dataset in [
            "Cora",
            "Citeseer",
            "Pubmed",
            "ogbn-arxiv",
            "Reddit",
        ]:
            if args.train_mode == "normal":
                args.init_mode = "kaiming_uniform"
                args.linear_sparsity = 0
                args.scheduler = "CustomCosineLR"
                args.enable_abs_comp = False
                args.dropout = 0.0
                # args.weight_decay = 0.0005
                # args.weight_l1 = 0.0005
            elif args.train_mode == "score_only":
                # args.bn_affine = False
                # args.type_norm = "None"
                # args.weight_decay = 0.0
                # args.weight_l1 = 0.0
                # args.dropout = 0.0
                args.scheduler = "CustomCosineLR"
            return args
        elif args.dataset in [
            "ogbg-molhiv",
            "ogbg-molbace",
            "ogbg-molpcba",
            "hep10k",
        ]:
            if args.train_mode == "normal":
                args.init_mode = "kaiming_uniform"
                args.linear_sparsity = 0
                args.scheduler = "CustomCosineLR"
                args.enable_abs_comp = False
                args.dropout = 0.5
                args.weight_decay = 0.0005
                args.weight_l1 = 0.0005
            elif args.train_mode == "score_only":
                # args.bn_affine = False
                # args.type_norm = "None"
                # args.weight_decay = 0.0
                # args.weight_l1 = 0.0
                # args.dropout = 0.5
                args.scheduler = "CustomCosineLR"
            return args

    def reset_dataset_dependent_parameters(self, args):
        if args.dataset == "Cora":
            args.num_feats = 1433
            args.num_classes = 7
            args.num_nodes = 2708
            # args.dropout = 0.6
            args.dropout = 0
            # args.type_norm = "None"
            # args.bn_affine = False
            args.weight_decay = 0
            args.weight_l1 = 5e-4
            args.activation = "relu"
            args.nm_decay = 0.0015
            args.lr = 0.01

        elif args.dataset == "Pubmed":
            args.num_feats = 500
            args.num_classes = 3
            args.num_nodes = 19717
            # args.dropout = 0.5
            args.dropout = 0
            # args.type_norm = "None"
            # args.bn_affine = False
            args.weight_decay = 0
            args.weight_l1 = 5e-4
            args.activation = "relu"
            args.nm_decay = 0.0015
            args.lr = 0.01

        elif args.dataset == "Citeseer":
            args.num_feats = 3703
            args.num_classes = 6
            args.num_nodes = 3327
            # args.dropout = 0.7
            args.dropout = 0
            # args.type_norm = "None"
            # args.bn_affine = False
            args.weight_decay = 0
            args.weight_l1 = 5e-4
            args.nm_decay = 0.00225
            args.activation = "relu"
            args.lr = 0.01

        elif args.dataset == "ogbn-arxiv":
            args.num_feats = 128
            args.num_classes = 40
            args.num_nodes = 169343
            args.dropout = 0.0
            args.weight_decay = 0.001
            args.weight_l1 = 0.0005
            args.epochs = 500
            args.lr = 0.01
            args.nm_decay = 0.0005
            args.type_norm = "batch"
            args.bn_affine = True

        elif args.dataset == "Reddit":
            args.num_feats = 602
            args.num_classes = 41
            args.num_nodes = 232965
            args.type_norm = "batch"
            args.bn_affine = True
            args.dropout = 0
            # args.dropout = 0.5
            # args.epochs = 400
            args.lr = 0.01
            args.nm_decay = 0.0015

        elif args.dataset == "ogbg-molhiv":
            args.num_feats = 9
            args.num_classes = 2
            # args.dropout = 0.5
            args.weight_l1 = 0.0
            args.weight_decay = 0.0005
            args.nm_decay = 0.0001
            # args.dim_hidden = 448

        elif args.dataset == "ogbg-molbace":
            args.num_feats = 9
            args.num_classes = 2
            # args.dropout = 0.5
            args.weight_l1 = 0.0
            args.weight_decay = 0.0005
            args.nm_decay = 0.0001
            # args.dim_hidden = 448

        elif args.dataset == "ogbg-molpcba":
            args.num_feats = 9
            args.num_classes = 2
            # args.dropout = 0.5
            args.weight_l1 = 0.0
            args.weight_decay = 0.0005
            args.nm_decay = 0.0001
            # args.dim_hidden = 448

        elif args.dataset == "hep10k":
            args.num_feats = 9
            args.num_classes = 2
            # args.dropout = 0.5
            args.weight_l1 = 0.0
            args.weight_decay = 0.0005
            args.nm_decay = 0.0001
            # args.dim_hidden = 448

        # ==============================================
        return args
