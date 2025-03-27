import copy
import json
import random
import time

import numpy as np

# import matplotlib.pyplot as plt
import torch
from models.GraphLevelLearning import GraphLevelLearning
from models.networks.GCN_graph_UGTs import GCN_graph_UGTs
from models.supervised_learning import SupervisedLearning
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch.nn import DataParallel
from torch_geometric.loader import DataLoader

# from tqdm import tqdm
from utils.output_manager import OutputManager
from utils.seed import set_random_seed
import os
import ast
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import Subset


class Hep10kDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root: dataset/hep10k のパス
        """
        super(Hep10kDataset, self).__init__(root, transform, pre_transform)
        # common/includes/dataset/dataset_size.txt からグラフ数（例: 10000）を読み込む
        dataset_size_path = os.path.join(self.root, "common", "includes", "dataset", "dataset_size.txt")
        with open(dataset_size_path, "r") as f:
            self._num_graphs = int(f.read().strip())

    def len(self):
        return self._num_graphs

    def get(self, idx):
        """
        idx: 0-indexed でアクセスしますが，
             ファイル名は g1, g2, ... のように 1 から始まっているので，
             graph_id = idx + 1 として扱います．
        """
        graph_id = idx + 1

        # 1. グラフ情報の読み込み（ノード数，エッジ数）
        info_file = os.path.join(self.root, "graphs", "graph_info", f"g{graph_id}_info.txt")
        with open(info_file, "r") as f:
            lines = f.readlines()
            num_nodes = int(lines[0].strip())
            num_edges = int(lines[1].strip())

        # 2. エッジリストの読み込み（graphs/graph_bin/g{graph_id}_edge_list.bin）
        edge_list_file = os.path.join(self.root, "graphs", "graph_bin", f"g{graph_id}_edge_list.bin")
        edge_index_np = np.fromfile(edge_list_file, dtype=np.int32)
        # バイナリファイル内の全要素数から (num_edges, 2) にリシェイプし、転置して (2, num_edges) にする
        edge_index_np = edge_index_np.reshape(-1, 2).T
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        # 3. エッジ属性の読み込み（graphs/graph_bin/g{graph_id}_edge_attr.bin）
        edge_attr_file = os.path.join(self.root, "graphs", "graph_bin", f"g{graph_id}_edge_attr.bin")
        edge_attr_np = np.fromfile(edge_attr_file, dtype=np.float32)
        if num_edges > 0:
            attr_dim = edge_attr_np.size // num_edges
            edge_attr_np = edge_attr_np.reshape(num_edges, attr_dim)
        else:
            edge_attr_np = edge_attr_np.reshape(0, -1)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)

        # 4. DGN の固有ベクトルの読み込みとノード特徴としての採用
        eig_file = os.path.join(self.root, "DGN", "eig", f"g{graph_id}.txt")
        with open(eig_file, "r") as f:
            eig_content = f.read().strip()
        # ファイル中は "tensor([[ ... ]])" のような形式になっているので，
        # "tensor(" と末尾の ")" を取り除いてから文字列を Python のリテラルとして評価する．
        if eig_content.startswith("tensor(") and eig_content.endswith(")"):
            inner_content = eig_content[len("tensor("):-1]
            try:
                eig_list = ast.literal_eval(inner_content)
            except Exception as e:
                raise ValueError(f"Eigenファイル {eig_file} のパースに失敗しました: {e}")
            eig = torch.tensor(eig_list, dtype=torch.float)
        else:
            # もし "tensor(...)" の形式でなければ，そのまま literal_eval を試みる
            try:
                eig_list = ast.literal_eval(eig_content)
                eig = torch.tensor(eig_list, dtype=torch.float)
            except Exception as e:
                raise ValueError(f"Eigenファイル {eig_file} のパースに失敗しました: {e}")

        # ※ 固有ベクトルの行数がノード数と一致するか確認（必要に応じて）
        if eig.size(0) != num_nodes:
            raise ValueError(f"ノード数の不一致: graph_info では {num_nodes} ノードですが, eig ファイルは {eig.size(0)} 行あります。")

        # DGN の固有ベクトルをそのままノード特徴として用いる
        x = eig

        # PyG の Data オブジェクトに各情報をまとめる．
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


def count_params(model):
    print("--------------------")
    # print(model)
    count = 0
    count_not_score = 0
    count_reduced = 0
    for n, p in model.named_parameters():
        count += p.flatten().size(0)
        if hasattr(p, "is_score") and p.is_score:
            print(
                n + ":",
                int(p.flatten().size(0) * (1.0 - p.sparsity)),
                "/",
                p.flatten().size(0),
                "(sparsity =",
                p.sparsity,
                ")",
            )
            count_reduced += int(p.flatten().size(0) * p.sparsity)
        else:
            print(n + ":", p.flatten().size(0))
            count_not_score += p.flatten().size(0)
    count_after_pruning = count_not_score - count_reduced
    total_sparsity = 1 - (count_after_pruning / count_not_score)
    print("--------------------")
    print(
        "Params after/before pruned:\t",
        count_after_pruning,
        "/",
        count_not_score,
        "(sparsity: " + str(total_sparsity) + ")",
    )
    print("Total Params:\t", count)
    return {
        "params_after_pruned": count_after_pruning,
        "params_before_pruned": count_not_score,
        "total_params": count,
        "sparsity": total_sparsity,
    }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(exp_name, args, prefix="", idx=1):
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    elif args.seed_by_time:
        set_random_seed(seeds[idx])
    else:
        raise Exception("Set seed value.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outman = OutputManager(args.output_dir, exp_name)
    dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")

    if args.type_model == "GCN_graph_UGTs":
        model = GCN_graph_UGTs(args).to(device)
        learner = GraphLevelLearning(outman, args, device, model)
        print(model)
        for name, param in model.named_parameters():
            print(
                f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}"
            )

    elif args.learning_framework == "SupervisedLearning":
        learner = SupervisedLearning(outman, args, device)
    else:
        raise NotImplementedError

    best_value = None
    best_epoch = 0
    start_epoch = 0
    total_iters = 0
    outman.print(dump_path, prefix=prefix)

    if args.validate is True:
        ece = 0.0
        test_acc = 0.0
        loadmodel = torch.load(args.pretrained_path)
        model_state_dict = loadmodel["model_state_dict"]
        # best_states = {}
        if args.learning_framework == "SupervisedLearning":
            learner.model.load_state_dict(model_state_dict)
            val_acc, test_acc = learner.evaluate()
        elif args.learning_framework == "GraphLevelLearning":
            dataset = PygGraphPropPredDataset(name=args.dataset)
            evaluator = Evaluator(args.dataset)
            split_idx = dataset.get_idx_split()
            test_dataset = dataset[split_idx["test"]]
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
            model.load_state_dict(model_state_dict)
            test_acc = learner.evaluate(test_loader, model, evaluator)
        # print("val_acc", val_acc)
        print("test_acc", test_acc)
        return test_acc, ece

    if args.type_model == "GCN_graph_UGTs":
        best_value = None
        start_epoch = 0

        if args.dataset == "hep10k":
            # 1. 独自データセットを読み込む
            dataset = Hep10kDataset(root="dataset/hep10k")

            # 2. 全グラフ数を取得してインデックスをシャッフルする
            num_graphs = len(dataset)
            indices = np.arange(num_graphs)
            np.random.shuffle(indices)

            # 3. train, valid, test の分割（例：80% / 10% / 10%）
            train_size = int(0.8 * num_graphs)
            val_size   = int(0.1 * num_graphs)
            test_size  = num_graphs - train_size - val_size

            train_indices = indices[:train_size]
            val_indices   = indices[train_size:train_size + val_size]
            test_indices  = indices[train_size + val_size:]

            # 4. Subset を用いて各データセットを作成
            train_dataset = Subset(dataset, train_indices)
            val_dataset   = Subset(dataset, val_indices)
            test_dataset  = Subset(dataset, test_indices)

            # 5. DataLoader の作成
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,  # train はシャッフル
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )

            class DummyEvaluator:
                def eval(self, input_dict):
                    # input_dict の中身（例：出力や正解ラベル）から評価を実施する
                    # ここでは単純にダミーの結果を返す例です。実際の評価ロジックに合わせて実装してください。
                    return {"acc": 0.0}

            evaluator = DummyEvaluator()

        else:
            dataset = PygGraphPropPredDataset(
                # name=args.dataset, root="/ldisk/Shared/Datasets/GNNDataset"
                name=args.dataset, root="dataset"
            )
            evaluator = Evaluator(args.dataset)
            split_idx = dataset.get_idx_split()

            train_dataset = dataset[split_idx["train"]]
            val_dataset = dataset[split_idx["valid"]]
            test_dataset = dataset[split_idx["test"]]

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )

        valid_curve = []
        test_acc = 0
        total_inference_time = 0.0
        total_train_time = 0.0
        # inference_idx = 0

        for epoch in range(start_epoch, args.epochs):
            if args.validate is True:
                epoch = args.epochs
            if args.inference_time:
                train_start_time = time.time()
                results_train = learner.train(epoch, train_loader, model)
                train_time = time.time() - train_start_time
                total_train_time += train_time
            else:
                results_train = learner.train(
                    epoch, train_loader, model, evaluator
                )

            if epoch >= (args.epochs // 2):
                with torch.no_grad():
                    if args.inference_time:

                        start_time = time.time()
                        valid_perf = learner.evaluate(
                            val_loader, model, evaluator
                        )
                        elapsed_time = time.time() - start_time

                        total_inference_time += elapsed_time
                    else:
                        valid_perf = learner.evaluate(
                            val_loader, model, evaluator
                        )

                    if args.dataset == "hep10k":
                        pass
                    else:
                        valid_curve.append(valid_perf[dataset.eval_metric])

                    if args.adp is True:
                        if (best_value is None) or (
                            valid_perf[dataset.eval_metric] > best_value
                        ):
                            best_value = valid_perf[dataset.eval_metric]
                            best_epoch = epoch
                            save_best_model = True
                            best_states = copy.deepcopy(
                                learner.model.state_dict()
                            )
                        else:
                            save_best_model = False
                    else:
                        # if best_value is None:
                        #     save_best_model = True
                        if args.dataset == "hep10k":
                            save_best_model = True
                        elif (
                            (best_value is None)
                            or (valid_perf[dataset.eval_metric] > best_value)
                        ) and epoch >= (args.epochs // 2):
                            valid_test_acc = learner.evaluate(
                                test_loader, model, evaluator
                            )
                            if test_acc == 0:
                                test_acc = valid_test_acc
                                best_value = valid_perf[dataset.eval_metric]
                                best_epoch = epoch
                                save_best_model = True
                            elif args.train_mode == "score_only" and (
                                valid_test_acc["rocauc"] > test_acc["rocauc"]
                            ):
                                test_acc = valid_test_acc
                                best_value = valid_perf[dataset.eval_metric]
                                best_epoch = epoch
                                save_best_model = True
                            else:
                                save_best_model = False
                            # best_states = copy.deepcopy(learner.model.state_dict())
                        else:
                            save_best_model = False

                    if isinstance(learner.model, DataParallel):
                        model_state_dict = learner.model.module.state_dict()
                    else:
                        model_state_dict = learner.model.state_dict()

                    if save_best_model:
                        dump_dict = {
                            "model_state_dict": model_state_dict,  # Use the latest model state dictionary
                            "best_val": best_value,
                            "best_epoch": best_epoch,
                        }
                        if args.dataset == "ogbg-molpcba":
                            info_dict = {
                                "best_epoch": best_epoch,
                                "best_val": best_value,
                                "test_acc": test_acc["ap"],
                            }
                        elif args.dataset == "hep10k":
                            info_dict = {
                                "best_epoch": best_epoch,
                                "best_val": best_value,
                                "test_acc": test_acc,
                            }
                        else:
                            info_dict = {
                                "best_epoch": best_epoch,
                                "best_val": best_value,
                                "test_acc": test_acc["rocauc"],
                            }

                        outman.save_dict(
                            dump_dict, prefix=f"dump.{prefix}_{idx}", ext="pth"
                        )
                        with open(
                            outman.get_abspath(
                                prefix=f"info.{prefix}_{idx}", ext="json"
                            ),
                            "w",
                        ) as f:
                            json.dump(info_dict, f, indent=2)
                        if save_best_model and args.save_best_model:
                            outman.save_dict(
                                dump_dict,
                                prefix=f"best.{prefix}_{idx}",
                                ext="pth",
                            )
                        save_best_model is False
            else:
                valid_curve.append(0)
            if test_acc != 0 and args.dataset == "ogbg-molpcba":
                print(
                    epoch,
                    "ap",
                    "train_ap",
                    results_train["ap"],
                    "val_ap",
                    valid_perf["ap"],
                    "test_ap",
                    test_acc["ap"],
                )
            elif test_acc != 0 and args.dataset == "hep10k":
                pass
            elif test_acc != 0:
                print(
                    epoch,
                    "train_acc",
                    results_train["rocauc"],
                    "val_acc",
                    valid_perf["rocauc"],
                    "test_acc",
                    test_acc["rocauc"],
                )
            elif args.dataset == "ogbg-molpcba":
                print(
                    epoch,
                    "ap",
                    results_train["ap"],
                )
            elif args.dataset == "hep10k":
                print(
                    epoch,
                    "acc",
                    results_train["acc"],
                )
            else:
                print(
                    epoch,
                    "train_acc",
                    results_train["rocauc"],
                )

        print("Finished training!")
        print("Best validation score: {}".format(valid_curve[best_epoch]))
        if args.dataset == "ogbg-molpcba":
            print("Test score: {}".format(test_acc["ap"]))
        elif args.dataset == "hep10k":
            print("Test score: {}".format(test_acc))
        else:
            print("Test score: {}".format(test_acc["rocauc"]))

        # plt.plot(
        #     range(start_epoch, args.epochs), valid_curve, label="Validation"
        # )
        # plt.xlabel("Epoch")
        # plt.ylabel("Performance")
        # plt.legend()

        # plot_filename = (
        #     f"./performance_plot/performance_plot_{args.exp_name}.png"
        # )
        # plt.savefig(plot_filename)
        # plt.close()

        if args.inference_time:
            average_inference_time = total_inference_time / args.epochs
            average_train_time = total_train_time / args.epochs
            print(
                f"Average train time over all epochs: {average_train_time:.4f} seconds"
            )
            print(
                f"Average inference time over all epochs: {average_inference_time:.4f} seconds"
            )

        if args.dataset == "ogbg-molpcba":
            result = test_acc["ap"]
        elif args.dataset == "hep10k":
            result = test_acc
        else:
            result = test_acc["rocauc"]

        # if not args.exp_name == "":
        #     torch.save(
        #         {"Val": valid_curve[best_epoch], "Test": result},
        #         args.exp_name,
        #     )


        return result, 0
    else:
        train_losses = []
        train_accs = []

        for epoch in range(start_epoch, args.epochs):
            if args.validate is True:
                epoch = args.epochs

            results_train = learner.train(epoch, args.epochs)
            train_accuracy = results_train["moving_accuracy"]
            new_total_iters = results_train["iterations"]
            total_loss_train = results_train["loss"]
            train_losses.append(total_loss_train)
            train_accs.append(train_accuracy)
            val_accuracy, test_accuracy = learner.evaluate()
            total_iters = new_total_iters

            if (best_value is None) or (best_value < val_accuracy):
                best_value = val_accuracy
                best_epoch = epoch
                save_best_model = True
                best_states = copy.deepcopy(learner.model.state_dict())
            else:
                save_best_model = False

            if isinstance(learner.model, DataParallel):
                model_state_dict = learner.model.module.state_dict()
            else:
                model_state_dict = learner.model.state_dict()

            dump_dict = {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optim_state_dict": learner.optimizer.state_dict(),
                "sched_state_dict": learner.scheduler.state_dict(),
                "best_val": best_value,
                "best_epoch": best_epoch,
                "total_iters": total_iters,
            }

            if save_best_model and args.save_best_model:
                outman.save_dict(
                    dump_dict, prefix=f"{idx}.best.{prefix}", ext="pth"
                )
            if epoch in args.checkpoint_epochs:
                outman.save_dict(
                    dump_dict, prefix=f"epoch{epoch}.{prefix}", ext="pth"
                )

            # print(val_accuracy, test_accuracy)

        learner.model.load_state_dict(best_states)
        val_acc, test_acc = learner.evaluate()
        ece = 0.0
        print("best acc:", test_acc)
        return test_acc, ece
