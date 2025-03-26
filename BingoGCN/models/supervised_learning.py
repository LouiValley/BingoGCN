import datetime
import math
import os
import random
import struct

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn.metrics as sk
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision.datasets
# import utils.datasets
from models.calibration import expected_calibration_error
from models.Dataloader import load_data, load_ogbn
from models.networks.dmGAT import dmGAT
from models.networks.dmGCN import dmGCN
from models.networks.dmgin_net import dmGINNet
from models.networks.FAdmGCN import FAdmGCN
from models.networks.GAT import GAT
from models.networks.GCN import GCN
from models.networks.gin_net import GINNet
from models.ood import *
from models.utils_gin import load_adj_raw_gin, load_data_gin

# from numpy import reshape
from ogb.nodeproppred import Evaluator
from sklearn.manifold import TSNE
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.datasets import Reddit
from torch_geometric.nn import GCNConv

# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_sparse import SparseTensor

# from torchvision.transforms import (
#     CenterCrop,
#     ColorJitter,
#     Compose,
#     Normalize,
#     Pad,
#     RandomHorizontalFlip,
#     RandomResizedCrop,
#     Resize,
#     ToTensor,
# )
from utils.schedulers import CustomCosineLR

# from tqdm import tqdm


# from utils.subset_dataset import SubsetDataset, random_split

# random seed


# random.seed(0)  # 可以替换0为您选择的任何数值

# np.random.seed(0)  # 可以替换0为您选择的任何数值
# torch.manual_seed(0)  # 可以替换0为您选择的任何数值

# # 如果您使用CUDA
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)  # 对所有GPU生效


def calculate_and_plot_sparsity(weight):
    """
    weight配列の各行におけるsparsity（0の割合）を計算し、
    その分布をヒストグラムとして描画して画像ファイルに保存する関数。
    """
    # sparsityの計算
    sparsity = np.mean(weight == 0, axis=1)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sparsity_distribution_{now}.png"
    # sparsityの分布をヒストグラムとして描画
    plt.figure(figsize=(10, 6))
    plt.hist(sparsity, bins=20, color="blue", edgecolor="black")
    plt.title("Sparsity Distribution")
    plt.xlabel("Sparsity")
    plt.ylabel("Frequency")

    # 画像として保存
    plt.savefig(filename)


def get_sparsity(sparsity, current_epoches, start_epoches, end_epoches):
    sparsity = sparsity - sparsity * (
        1
        - (current_epoches - start_epoches)
        * 1.0
        / (end_epoches - start_epoches)
    )
    return sparsity


class SupervisedLearning(object):
    def __init__(self, outman, args, device):
        self.args = args
        # print(self.args)
        self.outman = outman
        # self.cfg = cfg
        self.device = device
        # self.data_parallel = torch.cuda.is_available()
        self.data_parallel = False

        self.model_cfg = args.type_model
        if (
            self.model_cfg == "GIN"
            or self.model_cfg == "dmGIN"
            or self.model_cfg == "dgl_GAT"
        ):
            if self.args.dataset == "ogbn-arxiv":
                self.data, self.split_idx = load_ogbn(
                    self.args.dataset,
                    self.args.sampling,
                    self.args.samplingtype,
                    sparse_tensor=True,
                )

                self.idx_train = self.split_idx["train"]
                self.idx_test = self.split_idx["test"]
                self.idx_val = self.split_idx["valid"]
                print("idx train shape", self.idx_train.shape)
                adj = to_scipy_sparse_matrix(self.data.edge_index).tocoo()

                self.g = dgl.DGLGraph()
                node_num = self.data.x.size(0)
                class_num = self.data.y.numpy().max() + 1
                self.g.add_nodes(node_num)
                self.g.add_edges(adj.row, adj.col)

                self.g = self.g.to(device)
                self.features = self.data.x.to(device)
                self.labels = self.data.y.squeeze().to(device)
                self.g.adjacency_matrix = (
                    torch.tensor(adj.toarray()).to("cuda").float()
                )
                self.evaluator = Evaluator(name="ogbn-arxiv")
            else:

                adj, features, labels, idx_train, idx_val, idx_test = (
                    load_data_gin(args.dataset.lower())
                )
                adj = load_adj_raw_gin(args.dataset.lower())
                self.idx_train = idx_train
                self.idx_val = idx_val
                self.idx_test = idx_test
                node_num = features.size()[0]
                class_num = labels.numpy().max() + 1

                self.g = dgl.DGLGraph()
                self.g.add_nodes(node_num)
                adj = adj.tocoo()
                self.g.add_edges(adj.row, adj.col)
                self.g = self.g.to("cuda:0")
                self.features = features.to(self.device)
                self.labels = labels.to(self.device)
                self.g.adjacency_matrix = (
                    torch.tensor(adj.toarray()).to("cuda").float()
                )

        else:
            if self.args.dataset == "ogbn-arxiv":
                self.data, self.split_idx = load_ogbn(
                    self.args.dataset,
                    self.args.sampling,
                    self.args.samplingtype,
                    sparse_tensor=True,
                )
                self.data.to(self.device)
                self.idx_train = self.split_idx["train"]
                self.idx_test = self.split_idx["test"]
                self.idx_val = self.split_idx["valid"]
                self.evaluator = Evaluator(name="ogbn-arxiv")

                if self.args.only_train_data:
                    self.idx_train = self.idx_train.to(self.device)
                    self.data_train = self.data.subgraph(self.idx_train)

            elif self.args.dataset == "Reddit":
                # Model instantiation and training setup
                dataset = Reddit(
                    # root="/ldisk/Shared/Datasets/GNNDataset/Reddit"
                    root="dataset/Reddit"
                )
                self.data = dataset[0].to(self.device)

                # Convert edge_index to SparseTensor
                num_nodes = self.data.num_nodes
                row, col = self.data.edge_index
                self.data.edge_index = SparseTensor(
                    row=row, col=col, sparse_sizes=(num_nodes, num_nodes)
                )

            else:
                self.data = load_data(
                    self.args.dataset,
                    self.args.random_seed,
                    self.args.attack,
                    self.args.attack_eps,
                )
                if args.auroc:
                    self.data, class_num = get_ood_split(
                        self.data.cpu(), ood_frac_left_out_classes=0.4
                    )
                    self.args.num_classes = class_num
                    self.data.cuda()

                if self.args.only_train_data:
                    self.data_train = self.data.subgraph(self.data.train_mask)
                    # self.data_train.train_mask = self.data.train_mask[
                    #     self.data_train.ndata[dgl.NID]
                    # ]
                    self.data_train = self.data_train.to(self.device)

        self.model = self._get_model().to(self.device)  # define model
        print("--------------------")
        print(self.model)
        if self.model_cfg == "GCN":
            # print("the first 27 layers are fixed!")
            if self.args.num_layers == 32:  # I can't understand these codes
                for i in range(0, 5):
                    self.model.module.layers_GCN[
                        i
                    ].lin.weight_score.sparsity = self.args.linear_sparsity
        self.optimizer = self._get_optimizer(self.args.train_mode, self.model)
        # self.scheduler = self._get_scheduler()
        self.criterion = self._get_criterion()
        self.scheduler = self._get_scheduler()

    # ↓この実装は本当にL1の実装になっている？sigmoidなどの意味が不明
    def L1_norm(self):
        reg_loss = 0.0
        # ratio = 0.0
        # n = 0
        for param in self.model.parameters():
            if hasattr(param, "is_score") and param.is_score:
                # n += 1
                # print(param.shape)
                param = param.squeeze()
                assert len(param.shape) == 2 or len(param.shape) == 1
                # param=torch.sigmoid(param)
                reg_loss = reg_loss + torch.mean(torch.sigmoid(param))
        return reg_loss, None

    # 自分で実装をかえたL1_norm (2024-06-19)
    # def L1_norm(self):
    #     reg_loss = 0.0
    #     for param in self.model.parameters():
    #         if hasattr(param, "is_score") and param.is_score:
    #             param = param.squeeze()
    #             reg_loss = reg_loss + torch.sum(torch.abs(param))
    #     return reg_loss

    def train(self, epoch, total_iters):

        self.model.train()

        # add codes for multi-mask linear sparsity
        half_epochs = self.args.epochs / 2.0
        use_initial_sparsity = self.args.sparse_decay and epoch < half_epochs

        if self.args.enable_mask:
            sparsity = [
                (
                    get_sparsity(vs, epoch, 0, half_epochs)
                    if use_initial_sparsity
                    else vs
                )
                for vs in self.args.sparsity_list
            ]
        else:
            sparsity = (
                get_sparsity(self.args.linear_sparsity, epoch, 0, half_epochs)
                if use_initial_sparsity
                else self.args.linear_sparsity
            )

        # if self.args.enable_feat_pruning:
        #     x_sparsity = get_sparsity(self.args.featsparsity_ratio, epoch, 0, half_epochs)
        # else:
        #     x_sparsity = None

        results = []
        total_count = 0
        total_loss = 0.0
        correct = 0
        # iters_per_epoch = 1
        step_before_train = (
            hasattr(self.scheduler, "step_before_train")
            and self.scheduler.step_before_train
        )
        if step_before_train:
            try:
                self.scheduler.step()
            except:
                self.scheduler.step()
        for _ in range(1):

            # print("shape",self.data.edge_index.shape)
            if self.model_cfg == "GIN":
                outputs = self.model(
                    self.g, self.features, 0, 0, sparsity=sparsity
                )
                loss = self.criterion(
                    outputs[self.idx_train], self.labels[self.idx_train]
                )
                targets = self.labels[self.idx_train]
            elif self.model_cfg == "dmGIN":
                outputs = self.model(
                    self.g, self.features, 0, 0, sparsity=sparsity
                )
                loss = self.criterion(
                    outputs[self.idx_train], self.labels[self.idx_train]
                )
                targets = self.labels[self.idx_train]
            elif self.model_cfg == "dgl_GAT":
                outputs = self.model(self.g, self.features, sparsity=sparsity)
                loss = self.criterion(
                    outputs[self.idx_train], self.labels[self.idx_train]
                )
                targets = self.labels[self.idx_train]
            else:
                if (
                    self.model_cfg == "dmGCN"
                    or self.model_cfg == "dmGAT"
                    or self.model_cfg == "FAdmGCN"
                ):
                    adj_matrix = torch.zeros(
                        (self.data.num_nodes, self.data.num_nodes),
                        dtype=torch.float32,
                    )
                    adj_matrix[
                        self.data.edge_index[0], self.data.edge_index[1]
                    ] = 1

                    if self.args.spadj is True:
                        # 设置稀疏化的比例，例如30%
                        adjsparsity_ratio = self.args.adjsparsity_ratio
                        # 获取邻接矩阵中所有非零元素的位置
                        edges = adj_matrix.nonzero(as_tuple=True)
                        # 计算要移除的边数
                        num_edges_to_remove = int(
                            adjsparsity_ratio * len(edges[0])
                        )

                        # 随机选择要移除的边
                        edges_to_remove = np.random.choice(
                            len(edges[0]), num_edges_to_remove, replace=False
                        )
                        # curradj_sparsity = torch.sum(adj_matrix == 0).item() / adj_matrix.numel()
                        # print('before sparse adj sparsity is', curradj_sparsity)
                        # 对邻接矩阵进行稀疏化
                        for idx in edges_to_remove:
                            adj_matrix[edges[0][idx], edges[1][idx]] = 0
                        # curradj_sparsity = torch.sum(adj_matrix == 0).item() / adj_matrix.numel()
                        # print('after sparse adj sparsity is', curradj_sparsity)
                    if self.args.drawadj is True:
                        deg_inv_sqrt = (
                            adj_matrix.sum(dim=-1).clamp(min=1).pow(-0.5)
                        )
                        adj2 = (
                            deg_inv_sqrt.unsqueeze(-1)
                            * adj_matrix
                            * deg_inv_sqrt.unsqueeze(-2)
                        )
                        adj_np = adj2.to_dense().numpy()
                        # 创建一个 RGB 彩色图像，其中非零值为蓝色
                        height, width = adj_np.shape
                        image = (
                            np.ones((height, width, 3), dtype=np.uint8) * 100
                        )  # 初始化为灰白色（接近白色的灰色）
                        image[adj_np != 0] = [0, 255, 0]  # red
                        # 使用 matplotlib 保存图像
                        plt.imshow(image)
                        plt.axis("off")  # 关闭坐标轴
                        plt.savefig(
                            "adjacency_matrix11.png",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        adj_sparsity = (
                            torch.sum(adj2 == 0).item() / adj2.numel()
                        )
                        print("adj sparsity is", adj_sparsity)

                    outputs = self.model(
                        self.data.x,
                        adj_matrix,
                        sparsity=sparsity,
                        epoch=epoch,
                    )

                else:
                    if self.args.only_train_data:
                        outputs = self.model(
                            self.data_train.x,
                            self.data_train.edge_index,
                            sparsity=sparsity,
                            epoch=epoch,
                        ).to(self.device)
                        if self.args.dataset == "ogbn-arxiv":
                            targets = self.data.y.squeeze()[self.idx_train].to(
                                self.device
                            )
                            loss = self.criterion(
                                outputs.to(self.device), targets
                            )
                        else:
                            loss = self.criterion(
                                outputs[self.data_train.train_mask],
                                self.data_train.y[self.data_train.train_mask],
                            )
                            targets = self.data_train.y[
                                self.data_train.train_mask
                            ]
                    else:
                        outputs = self.model(
                            self.data.x,
                            self.data.edge_index,
                            sparsity=sparsity,
                            epoch=epoch,
                        ).to(self.device)
                        if self.args.dataset == "ogbn-arxiv":
                            targets = self.data.y.squeeze()[self.idx_train].to(
                                self.device
                            )
                            loss = self.criterion(
                                outputs[self.idx_train].to(self.device),
                                targets,
                            )
                        elif self.args.dataset == "Reddit":
                            loss = self.criterion(
                                outputs[self.data.train_mask].to(self.device),
                                self.data.y[self.data.train_mask].to(
                                    self.device
                                ),
                            )
                        else:
                            targets = self.data.y[self.data.train_mask].to(
                                self.device
                            )
                            loss = self.criterion(
                                outputs[self.data.train_mask], targets
                            )

            # np.savetxt(
            #     "outputs.txt",
            #     outputs.detach().cpu().numpy().flatten(),
            #     fmt="%f",
            # )

            if self.args.train_mode == "score_only":
                L1_loss, _ = self.L1_norm()
            else:
                L1_loss = torch.tensor(0.0).to(outputs.device)
            loss = loss + self.args.weight_l1 * L1_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.args.dataset == "ogbn-arxiv":
                if self.args.only_train_data:
                    _, predicted = outputs.max(1)
                else:
                    _, predicted = outputs[self.idx_train].max(1)
            elif self.args.dataset == "Reddit":
                predicted = outputs.argmax(dim=1)

            else:
                if (
                    self.model_cfg == "GIN"
                    or self.model_cfg == "dmGIN"
                    or self.model_cfg == "dgl_GAT"
                ):
                    _, predicted = outputs[self.idx_train].max(1)
                else:
                    if self.args.only_train_data:
                        _, predicted = outputs.max(1)
                    else:
                        _, predicted = outputs[self.data.train_mask].max(1)

            if self.args.dataset == "ogbn-arxiv":
                total_count += (
                    self.data.y[self.idx_train].to(self.device).size(0)
                )
            else:
                total_count += (
                    self.data.y[self.data.train_mask].to(self.device).size(0)
                )

            if self.args.dataset == "Reddit":
                correct = (
                    predicted[self.data.test_mask]
                    .eq(self.data.y[self.data.test_mask])
                    .sum()
                    .item()
                )
                self.model.eval()
                mean_loss = loss.item()
                # print("loss",mean_loss,"acc",correct / total_count)
                results.append(
                    {
                        "mean_loss": mean_loss,
                    }
                )

                total_loss += mean_loss
                total_iters += 1

                return {
                    "iterations": total_iters,
                    "per_iteration": results,
                    "loss": total_loss / total_count,
                    "moving_accuracy": correct
                    / self.data.test_mask.sum().item(),
                    "L1_loss": L1_loss.item(),
                }
            else:
                correct += predicted.eq(targets).sum().item()
            mean_loss = loss.item()
            # print("loss",mean_loss,"acc",correct / total_count)
            results.append(
                {
                    "mean_loss": mean_loss,
                }
            )

            total_loss += mean_loss
            total_iters += 1

        if not step_before_train:
            try:
                self.scheduler.step()
            except:
                self.scheduler.step()

        # print(
        #     predicted.eq(targets).sum().item()
        #     / (self.data.y[self.data.train_mask].to(self.device).size(0))
        # )
        self.model.eval()
        # print("Loss: ", total_loss / total_count)
        # print("Train Acc: ", correct / total_count)
        # print(correct / total_count)
        # print(L1_loss.item())
        return {
            "iterations": total_iters,
            "per_iteration": results,
            "loss": total_loss / total_count,
            "moving_accuracy": correct / total_count,
            "L1_loss": L1_loss.item(),
        }

    def get_ece(self):
        self.model.eval()
        if self.model_cfg == "GIN":
            labels = self.labels
            logits = self.model(self.g, self.features, 0, 0)
            yp = torch.softmax(logits, -1)
            ece = expected_calibration_error(
                labels[self.idx_test].cpu().detach().numpy(),
                yp[self.idx_test].cpu().detach().numpy(),
            )
            # _,indices=logits.max(1)
        else:
            labels = self.data.y
            logits = self.model(self.data.x, self.data.edge_index)
            # _, indices = torch.max(logits, dim=1)
            yp = torch.softmax(logits, -1)
            ece = expected_calibration_error(
                labels[self.data.test_mask].cpu().detach().numpy(),
                yp[self.data.test_mask].cpu().detach().numpy(),
            )
        return ece

    def plot_tsne(self, path):
        file_path = (
            path
            + self.args.dataset
            + "_"
            + self.model_cfg
            + "_"
            + str(self.args.random_seed)
            + "_"
            + str(self.args.linear_sparsity)
            + "_"
            + self.args.train_mode
            + "_"
            + str(self.args.num_layers)
            + ".jpg"
        )
        self.model.eval()
        sns.set_style("white")
        tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=50)
        if self.model_cfg == "GIN":
            labels = self.labels
            logits = self.model(self.g, self.features, 0, 0)
            z = tsne.fit_transform(
                logits[self.idx_test].cpu().detach().numpy()
            )
            y = labels[self.idx_test].cpu().detach().numpy()

            # yp=torch.softmax(logits,-1)
            # ece=expected_calibration_error(labels[self.idx_test].cpu().detach().numpy(),yp[self.idx_test].cpu().detach().numpy())
            # _,indices=logits.max(1)
        else:
            labels = self.data.y
            logits = self.model(self.data.x, self.data.edge_index)
            z = tsne.fit_transform(
                logits[self.data.test_mask].cpu().detach().squeeze().numpy()
            )
            y = labels[self.data.test_mask].cpu().detach().squeeze().numpy()
        df = pd.DataFrame()
        # print("y shape",y.shape,"z shape",z.shape)
        # df['y']=y
        # df['comp-1']=z[:,0]
        # df['comp-2']=z[:,1]
        length = int(z[:, 0].shape[0] * 1)
        all_len = z[:, 0].shape[0]

        indexes = np.arange(all_len)
        np.random.shuffle(indexes)
        samples_indexes = indexes[:length]
        # print("z shape",z.shape)
        x_samples = z[:, 0][samples_indexes]
        y_samples = z[:, 1][samples_indexes]
        hue_samples = y[samples_indexes].tolist()

        df["y"] = y[samples_indexes]
        df["comp-1"] = x_samples
        df["comp-2"] = y_samples
        y_set = set(y[samples_indexes])
        print(len(list(y_set)))
        # np.save(file_path[:-4]+"_z.npy",logits[self.idx_test].cpu().detach().squeeze().numpy())
        # np.save(file_path[:-4]+"_y.npy",y)
        # print("x shape",x_samples.shape)
        # print("y shape",y_samples.shape)
        # print(len(hue_samples ))

        fig = sns.scatterplot(
            x=x_samples,
            y=y_samples,
            hue=hue_samples,
            palette=sns.color_palette(n_colors=len(list(y_set))),
            data=df,
            s=100,
            legend=False,
            linewidth=0.5,
        )
        fig.set(xticklabels=[], yticklabels=[])
        fig.set(xlabel=None, ylabel=None)
        fig.tick_params(bottom=False, left=False, pad=0)
        sns.despine(top=True, right=True, left=True, bottom=True, trim=True)
        scatter = fig.get_figure()
        scatter.tight_layout()
        scatter.savefig(
            file_path, pad_inches=0.0, dpi=600, bbox_inches="tight"
        )

    def get_roc(self):

        self.model.eval()
        if self.model_cfg == "GIN":
            labels = self.labels
            logits = self.model(self.g, self.features, 0, 0)
            logits = torch.softmax(logits, dim=-1)
            # _,indices=logits.max(1)
            # print("total acc",indices.eq(self.labels).sum().item()/labels.size(0))
            # val_idx=torch.tensor(self.idx_val).long()
            # test_idx=torch.tensor(self.idx_test).long()
            # train_idx=torch.tensor(self.idx_train).long()
        else:
            labels = self.data.y
            logits = self.model(self.data.x, self.data.edge_index)
            logits = torch.softmax(logits, dim=-1)
            # _, indices = torch.max(logits, dim=1)

        # in distribution
        ind_scores, _ = logits[self.data.id_test_mask].max(dim=1)
        ind_scores = ind_scores.cpu().detach().numpy()
        ind_labels = np.zeros(ind_scores.shape[0])
        ind_scores = ind_scores * -1
        # ind_scores = np.max(y_pred_ind, 1)

        # out of distribution
        # y_pred_ood, _ = extract_prediction(out_loader, model, args)
        ood_scores, _ = logits[self.data.ood_test_mask].max(dim=1)
        ood_scores = ood_scores.cpu().detach().numpy()
        ood_labels = np.ones(ood_scores.shape[0])
        # ood_scores = np.max(y_pred_ood, 1)
        ood_scores = ood_scores * -1

        labels = np.concatenate([ind_labels, ood_labels])
        scores = np.concatenate([ind_scores, ood_scores])

        auroc = sk.roc_auc_score(labels, scores)
        print("* AUROC = {}".format(auroc))
        return auroc

    def evaluate(self):
        # print("id val",id(self.model))
        self.model.eval()
        if self.model_cfg == "GIN" or self.model_cfg == "dmGIN":
            labels = self.labels
            logits = self.model(self.g, self.features, 0, 0)
            _, indices = logits.max(1)
        elif self.model_cfg == "dgl_GAT":
            labels = self.labels
            logits = self.model(self.g, self.features)
            _, indices = logits.max(1)
        else:
            labels = self.data.y

            if (
                self.model_cfg == "dmGCN"
                or self.model_cfg == "dmGAT"
                or self.model_cfg == "FAdmGCN"
            ):
                adj_matrix = torch.zeros(
                    (self.data.num_nodes, self.data.num_nodes),
                    dtype=torch.float32,
                )
                adj_matrix[
                    self.data.edge_index[0], self.data.edge_index[1]
                ] = 1
                if self.args.spadj is True:
                    # 设置稀疏化的比例，例如30%
                    adjsparsity_ratio = self.args.adjsparsity_ratio
                    # 获取邻接矩阵中所有非零元素的位置
                    edges = adj_matrix.nonzero(as_tuple=True)
                    # 计算要移除的边数
                    num_edges_to_remove = int(
                        adjsparsity_ratio * len(edges[0])
                    )
                    # 随机选择要移除的边
                    edges_to_remove = np.random.choice(
                        len(edges[0]), num_edges_to_remove, replace=False
                    )
                    # curradj_sparsity = torch.sum(adj_matrix == 0).item() / adj_matrix.numel()
                    # print('before sparse adj sparsity is', curradj_sparsity)
                    # 对邻接矩阵进行稀疏化
                    for idx in edges_to_remove:
                        adj_matrix[edges[0][idx], edges[1][idx]] = 0

                logits = self.model(self.data.x, adj_matrix)
                _, indices = torch.max(logits, dim=1)

            else:
                if self.args.flowgnn_debug:
                    os.makedirs("./pretrained_model/Input/", exist_ok=True)
                    # part_edge_index = torch.tensor(
                    #     [[0, 2, 3, 5, 1, 8, 7, 1], [1, 1, 4, 0, 2, 3, 5, 4]]
                    # ).cuda()  # you create by yourself
                    # part_data_x = self.data.x[0:9]  # use load x

                    part_data_x = self.data.x
                    part_edge_index = self.data.edge_index

                    csr_sparse_tensor = part_data_x.to_sparse()
                    indices_s = csr_sparse_tensor.indices().cpu().numpy()
                    values = csr_sparse_tensor.values().cpu().numpy()
                    csr_matrix = scipy.sparse.csr_matrix(
                        (values, (indices_s[0], indices_s[1])),
                        shape=csr_sparse_tensor.shape,
                    )
                    csr_data = csr_matrix.data
                    csr_indices = csr_matrix.indices
                    csr_indptr = csr_matrix.indptr
                    combined_csr_data_indices = np.row_stack(
                        (csr_indices, csr_data)
                    )
                    print(
                        "Data & csr_indices:", combined_csr_data_indices.shape
                    )
                    print("csr_indptr len:", len(csr_indptr.data))

                    with open(
                        "./pretrained_model/Input/g1_csr_value.bin",
                        "wb",
                    ) as f:
                        f.write(csr_data.tobytes())

                    with open(
                        "./pretrained_model/Input/g1_csr_indice.bin",
                        "wb",
                    ) as f:
                        f.write(csr_indices.tobytes())

                    with open(
                        "./pretrained_model/Input/g1_csr_indptr.bin",
                        "wb",
                    ) as f:
                        f.write(csr_indptr.tobytes())

                    flattened_csr_value = csr_data.ravel(order="F")
                    flattened_csr_indices = csr_indices.ravel(order="F")
                    flattened_csr_indptr = csr_indptr.ravel(order="F")
                    np.savetxt(
                        "./pretrained_model/Input/csr_value_flatten.txt",
                        flattened_csr_value,
                        fmt="%f",
                        delimiter=",",
                    )
                    np.savetxt(
                        "./pretrained_model/Input/csr_indices_flatten.txt",
                        flattened_csr_indices,
                        fmt="%d",
                        delimiter=",",
                    )
                    np.savetxt(
                        "./pretrained_model/Input/csr_indptr_flatten.txt",
                        flattened_csr_indptr,
                        fmt="%d",
                        delimiter=",",
                    )

                    # organize edge_list
                    bin_edge_index = part_edge_index.cpu()
                    bin_edges = [
                        {"u": int(u), "v": int(v)}
                        for u, v in bin_edge_index.t().numpy()
                    ]
                    os.makedirs("./pretrained_model/Graph_bin/", exist_ok=True)

                    smoothed_edges = []
                    for edge in bin_edges:
                        smoothed_edges.append(edge["u"])
                        smoothed_edges.append(edge["v"])
                    smoothed_edges = np.array(smoothed_edges, dtype=np.int32)
                    # to bin file
                    with open(
                        "./pretrained_model/Graph_bin/g1_edge_list.bin",
                        "wb",
                    ) as f:
                        f.write(smoothed_edges.tobytes())

                    # with open(
                    #     "./pretrained_model/Graph_bin/g1_node_feature.bin",
                    #     "wb",
                    # ) as f:
                    #     for edge in bin_edges:
                    #         #  u->v
                    #         f.write(struct.pack("ii", edge["u"], edge["v"]))

                    x = part_data_x.cpu().numpy()
                    with open(
                        "./pretrained_model/Graph_bin/g1_node_feature.bin",
                        "wb",
                    ) as f:
                        f.write(x.tobytes())

                    num_nodes = len(part_data_x.cpu().numpy())
                    # エッジ数: smoothed_edges の半分 (各エッジはu,vで一組とカウント)
                    num_edges = len(smoothed_edges) // 2

                    # ディレクトリの作成
                    os.makedirs(
                        "./pretrained_model/graph_info/", exist_ok=True
                    )

                    # g1_info.txt に情報を書き込む
                    with open(
                        "./pretrained_model/graph_info/g1_info.txt", "w"
                    ) as f:
                        f.write(f"{num_nodes}\n")
                        f.write(f"{num_edges}\n")

                    logits = self.model(part_data_x, part_edge_index)

                else:
                    logits = self.model(self.data.x, self.data.edge_index)
                _, indices = torch.max(logits, dim=1)
        if self.args.dataset == "ogbn-arxiv":
            if self.model_cfg == "GIN" or self.model_cfg == "dgl_GAT":
                y_pred = logits.argmax(dim=-1, keepdim=True)
                acc_val = self.evaluator.eval(
                    {
                        "y_true": self.data.y.squeeze().unsqueeze(-1)[
                            self.idx_val
                        ],
                        "y_pred": y_pred[self.idx_val],
                    }
                )["acc"]
                acc_test = self.evaluator.eval(
                    {
                        "y_true": self.data.y.squeeze().unsqueeze(-1)[
                            self.idx_test
                        ],
                        "y_pred": y_pred[self.idx_test],
                    }
                )["acc"]
            else:
                y_pred = logits.argmax(dim=-1, keepdim=True)
                acc_val = self.evaluator.eval(
                    {
                        "y_true": self.data.y.squeeze().unsqueeze(-1)[
                            self.split_idx["valid"]
                        ],
                        "y_pred": y_pred[self.split_idx["valid"]],
                    }
                )["acc"]
                acc_test = self.evaluator.eval(
                    {
                        "y_true": self.data.y.squeeze().unsqueeze(-1)[
                            self.split_idx["test"]
                        ],
                        "y_pred": y_pred[self.split_idx["test"]],
                    }
                )["acc"]
        else:
            if self.model_cfg == "GIN" or self.model_cfg == "dmGIN":
                correct_val = torch.sum(
                    indices[self.idx_val] == labels[self.idx_val]
                )
                correct_test = torch.sum(
                    indices[self.idx_test] == labels[self.idx_test]
                )
                correct_train = torch.sum(
                    indices[self.idx_train] == labels[self.idx_train]
                )
                acc_train = correct_train.item() * 1.0 / len(self.idx_train)
                acc_val = correct_val.item() * 1.0 / len(self.idx_val)
                acc_test = correct_test.item() * 1.0 / len(self.idx_test)

            else:
                val_idx = self.data.val_mask
                test_idx = self.data.test_mask
                indices = indices.to(self.device)
                correct_val = torch.sum(indices[val_idx] == labels[val_idx])
                correct_test = torch.sum(indices[test_idx] == labels[test_idx])
                # correct_train=torch.sum(indices[train_idx]==labels[train_idx])
                # acc_train=correct_train.item()*1.0/train_idx.sum().item()
                acc_val = correct_val.item() * 1.0 / val_idx.sum().item()
                acc_test = correct_test.item() * 1.0 / test_idx.sum().item()

        return acc_val, acc_test

    def _get_model(self, model_cfg=None):
        if model_cfg is None:
            model_cfg = self.model_cfg

        if model_cfg == "GCN":
            model = GCN(self.args)
        elif model_cfg == "dmGCN":
            print("this is dmGCN for function model")
            model = dmGCN(self.args)
        elif model_cfg == "FAdmGCN":
            print("this is dmGCN for function model")
            model = FAdmGCN(self.args)
        elif model_cfg == "dmGAT":
            print("this is dmGAT for function model")
            model = dmGAT(self.args)
        elif model_cfg == "GAT":
            # model=GAT(self.args.num_feats,self.args.dim_hidden,self.args.num_layers)
            model = GAT(self.args)
        elif (
            model_cfg == "GIN"
            or model_cfg == "dgl_GAT"
            or model_cfg == "dmGIN"
        ):
            # model=GIN(self.args.num_feats,self.args.dim_hidden,self.args.num_layers)
            # model=GIN(self.args.num_feats,self.args.dim_hidden,self.args.num_layers,args=self.args)
            if model_cfg == "dgl_GAT":
                model = GATNet(self.args, self.g)
            elif model_cfg == "dmGIN":
                model = dmGINNet(self.args, self.g)
            else:
                model = GINNet(self.args, self.g)

        elif model_cfg == "SGC":
            model = SGC(self.args)
        else:
            raise NotImplementedError

        if self.data_parallel:
            gpu_ids = list(range(self.args.num_gpus))
            return DataParallel(model)
        else:
            return model

    def _get_optimizer(self, mode, model):
        # print("mode",mode)
        optim_name = self.args.optimizer
        # for name, param in model.named_parameters():
        #    print(name)
        if mode == "score_only":
            lr = self.args.lr
            weight_decay = self.args.weight_decay

            params = [
                param
                for param in model.parameters()
                if (hasattr(param, "is_score") and param.is_score)
            ]
            # print("update params")
            # for name, param in model.named_parameters():
            #     if hasattr(param, "is_score") and param.is_score:
            #         print(name)

            return self._new_optimizer(optim_name, params, lr, weight_decay)

        elif mode == "normal":
            lr = self.args.lr
            weight_decay = self.args.weight_decay
            params = [
                param
                for param in self.model.parameters()
                if not (hasattr(param, "is_score") and param.is_score)
            ]
            # print(params)
            return self._new_optimizer(optim_name, params, lr, weight_decay)
        else:
            raise NotImplementedError

    def _get_criterion(self):
        if self.args.dataset == "Reddit":
            return torch.nn.NLLLoss()
        else:
            return nn.CrossEntropyLoss()

    def _new_optimizer(self, name, params, lr, weight_decay, momentum=0.9):
        if name == "Adam":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == "SGD":
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=self.args.sgd_momentum,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError

    def _get_scheduler(self):
        class null_scheduler(object):
            def __init__(self, *args, **kwargs):
                return

            def step(self, *args, **kwargs):
                return

            def state_dict(self):
                return {}

            def load_state_dict(self, dic):
                return

        class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
            def __init__(
                self,
                optimizer: torch.optim.Optimizer,
                warmup_epochs: int,
                max_epochs: int,
                warmup_start_lr: float = 0.00001,
                eta_min: float = 0.00001,
                last_epoch: int = -1,
            ):
                """
                Args:
                    optimizer (torch.optim.Optimizer):
                        最適化手法インスタンス
                    warmup_epochs (int):
                        linear warmupを行うepoch数
                    max_epochs (int):
                        cosine曲線の終了に用いる 学習のepoch数
                    warmup_start_lr (float):
                        linear warmup 0 epoch目の学習率
                    eta_min (float):
                        cosine曲線の下限
                    last_epoch (int):
                        cosine曲線の位相オフセット
                学習率をmax_epochsに至るまでコサイン曲線に沿ってスケジュールする
                epoch 0からwarmup_epochsまでの学習曲線は線形warmupがかかる
                https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html
                """
                self.warmup_epochs = warmup_epochs
                self.max_epochs = max_epochs
                self.warmup_start_lr = warmup_start_lr
                self.eta_min = eta_min
                super().__init__(optimizer, last_epoch)
                return None

            def get_lr(self):
                if self.last_epoch == 0:
                    return [self.warmup_start_lr] * len(self.base_lrs)
                if self.last_epoch < self.warmup_epochs:
                    return [
                        group["lr"]
                        + (base_lr - self.warmup_start_lr)
                        / (self.warmup_epochs - 1)
                        for base_lr, group in zip(
                            self.base_lrs, self.optimizer.param_groups
                        )
                    ]
                if self.last_epoch == self.warmup_epochs:
                    return self.base_lrs
                if (self.last_epoch - 1 - self.max_epochs) % (
                    2 * (self.max_epochs - self.warmup_epochs)
                ) == 0:
                    return [
                        group["lr"]
                        + (base_lr - self.eta_min)
                        * (
                            1
                            - math.cos(
                                math.pi
                                / (self.max_epochs - self.warmup_epochs)
                            )
                        )
                        / 2
                        for base_lr, group in zip(
                            self.base_lrs, self.optimizer.param_groups
                        )
                    ]

                return [
                    (
                        1
                        + math.cos(
                            math.pi
                            * (self.last_epoch - self.warmup_epochs)
                            / (self.max_epochs - self.warmup_epochs)
                        )
                    )
                    / (
                        1
                        + math.cos(
                            math.pi
                            * (self.last_epoch - self.warmup_epochs - 1)
                            / (self.max_epochs - self.warmup_epochs)
                        )
                    )
                    * (group["lr"] - self.eta_min)
                    + self.eta_min
                    for group in self.optimizer.param_groups
                ]

        if self.args.scheduler is None:
            return null_scheduler()
        elif self.args.scheduler == "CustomCosineLR":
            total_epoch = self.args.epochs
            init_lr = self.args.lr
            # warmup_epochs = self.args.warmup_epochs
            # ft_epochs = self.args.finetuning_epochs
            # ft_lr = self.args.finetuning_lr
            return CosineAnnealingLR(
                self.optimizer,
                max_epochs=total_epoch,
                warmup_epochs=total_epoch**0.03,
                warmup_start_lr=init_lr,
                eta_min=0.00001,
            )
        elif self.args.scheduler == "MultiStepLR":
            return MultiStepLR(
                self.optimizer,
                milestones=self.args.lr_milestones,
                gamma=self.args.multisteplr_gamma,
            )
        else:
            raise NotImplementedError
